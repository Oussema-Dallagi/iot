import 'dotenv/config';
import express from 'express';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import multer from 'multer';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const PORT = Number(process.env.PORT) || 3000;
const app = express();

app.use(express.json({ limit: '10mb' }));

const uploadDir = path.join(os.tmpdir(), 'cadence-uploads');
const upload = multer({
  dest: uploadDir,
  limits: {
    fileSize: 250 * 1024 * 1024, // 250 MB video limit
  },
});

function formatServerError(error: unknown, fallback: string): string {
  const raw = error instanceof Error ? error.message.trim() : '';
  if (!raw) return fallback;
  if (raw.startsWith('{')) {
    try {
      const parsed = JSON.parse(raw) as {
        error?: { message?: unknown } | string;
        message?: unknown;
      };
      if (
        parsed.error &&
        typeof parsed.error === 'object' &&
        typeof parsed.error.message === 'string' &&
        parsed.error.message.trim()
      ) {
        return parsed.error.message.trim();
      }
      if (typeof parsed.error === 'string' && parsed.error.trim()) {
        return parsed.error.trim();
      }
      if (typeof parsed.message === 'string' && parsed.message.trim()) {
        return parsed.message.trim();
      }
    } catch {
      // Keep raw string if not valid JSON
    }
  }
  return raw;
}

function parseGeminiJsonObject(
  rawText: string | undefined,
  contextLabel: string
): Record<string, unknown> {
  const trimmed = (rawText ?? '').trim();
  if (!trimmed) {
    throw new Error(`Gemini returned an empty response while ${contextLabel}.`);
  }
  const stripped = trimmed
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  return JSON.parse(stripped) as Record<string, unknown>;
}

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured on the server.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function waitForFileActive(ai: GoogleGenAI, fileName: string) {
  for (let attempt = 0; attempt < 45; attempt += 1) {
    const fileState = await ai.files.get({ name: fileName });
    if (fileState.state === 'ACTIVE') {
      return fileState;
    }
    if (fileState.state === 'FAILED') {
      throw new Error('Video processing failed on Gemini File API.');
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error('Timed out waiting for uploaded video to finish processing.');
}

interface CachedParticipantPhoto {
  mimeType: string;
  data: string;
}

interface LiveSessionCacheEntry {
  title: string;
  participants: Array<{ id: string; name: string; role?: string }>;
  photos: Array<CachedParticipantPhoto | null>;
  createdAt: number;
}

const liveSessionCache = new Map<string, LiveSessionCacheEntry>();

function pruneStaleLiveSessions() {
  const cutoff = Date.now() - 2 * 60 * 60 * 1000;
  for (const [id, entry] of liveSessionCache.entries()) {
    if (entry.createdAt < cutoff) {
      liveSessionCache.delete(id);
    }
  }
}

const TRANSCRIPTION_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    durationSec: { type: Type.INTEGER },
    transcript: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING },
          startSec: { type: Type.INTEGER },
          endSec: { type: Type.INTEGER },
          speakerId: { type: Type.STRING },
          speakerName: { type: Type.STRING },
          text: { type: Type.STRING },
          highlightedPhrase: { type: Type.STRING },
          emotion: { type: Type.STRING },
          facialCue: { type: Type.STRING },
          emotionIntensity: { type: Type.INTEGER },
        },
        required: [
          'id',
          'startSec',
          'endSec',
          'speakerId',
          'speakerName',
          'text',
          'emotion',
          'facialCue',
          'emotionIntensity',
        ],
      },
    },
    actionItems: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING },
          task: { type: Type.STRING },
          assigneeId: { type: Type.STRING },
          assigneeName: { type: Type.STRING },
          priority: { type: Type.STRING },
          dueDate: { type: Type.STRING },
          sourceEntryId: { type: Type.STRING },
          timestampSec: { type: Type.INTEGER },
        },
        required: [
          'id',
          'task',
          'assigneeId',
          'assigneeName',
          'priority',
          'dueDate',
          'sourceEntryId',
          'timestampSec',
        ],
      },
    },
  },
  required: ['durationSec', 'transcript', 'actionItems'],
};

const ALLOWED_EMOTIONS = new Set([
  'Confident',
  'Focused',
  'Enthusiastic',
  'Concerned',
  'Hesitant',
  'Frustrated',
  'Supportive',
  'Neutral',
]);

function normalizeTranscriptionPayload(
  parsed: Record<string, unknown>,
  participants: Array<{ id: string; name: string; role?: string }>
) {
  const actionItems = (
    Array.isArray(parsed.actionItems) ? parsed.actionItems : []
  ).map((item: Record<string, unknown>, idx: number) => ({
    id: String(item.id || `act-${idx + 1}`),
    task: String(item.task || ''),
    assigneeId: String(item.assigneeId || participants[0]?.id || 'unknown'),
    assigneeName: String(
      item.assigneeName || participants[0]?.name || 'Unregistered Speaker'
    ),
    priority:
      item.priority === 'High' || item.priority === 'Low'
        ? item.priority
        : 'Medium',
    dueDate: String(item.dueDate || 'This week'),
    sourceEntryId: String(item.sourceEntryId || ''),
    timestampSec: Number(item.timestampSec || 0),
    completed: false,
  }));

  const actionBySourceId = new Map<string, string>();
  for (const act of actionItems) {
    if (act.sourceEntryId && !actionBySourceId.has(act.sourceEntryId)) {
      actionBySourceId.set(act.sourceEntryId, act.id);
    }
  }

  const transcript = (
    Array.isArray(parsed.transcript) ? parsed.transcript : []
  ).map((entry: Record<string, unknown>, idx: number) => {
    const id = String(entry.id || `tr-${idx + 1}`);
    const emotionRaw = String(entry.emotion || 'Neutral');
    const emotion = ALLOWED_EMOTIONS.has(emotionRaw) ? emotionRaw : 'Neutral';
    return {
      id,
      startSec: Number(entry.startSec || 0),
      endSec: Number(entry.endSec || 0),
      speakerId: String(entry.speakerId || 'unknown'),
      speakerName: String(entry.speakerName || 'Unregistered Speaker'),
      text: String(entry.text || ''),
      highlightedPhrase: entry.highlightedPhrase
        ? String(entry.highlightedPhrase)
        : undefined,
      actionItemId: actionBySourceId.get(id),
      emotion,
      facialCue: String(entry.facialCue || 'Observed speaking on camera'),
      emotionIntensity: Math.min(
        100,
        Math.max(10, Number(entry.emotionIntensity || 70))
      ),
    };
  });

  return {
    durationSec: Number(parsed.durationSec || 6),
    transcript,
    actionItems,
  };
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

app.post(
  '/api/meetings/live-session/init',
  upload.fields([{ name: 'participantPhotos', maxCount: 24 }]),
  async (req, res) => {
    const files = req.files as
      | { [fieldname: string]: Express.Multer.File[] }
      | undefined;
    const photoFiles = files?.participantPhotos ?? [];
    const tempPaths = photoFiles.map((pf) => pf.path);

    try {
      pruneStaleLiveSessions();
      const title = String(req.body?.title ?? 'Live Meeting').trim() || 'Live Meeting';
      const participants: Array<{ id: string; name: string; role?: string }> =
        JSON.parse(String(req.body?.participants ?? '[]'));

      if (!Array.isArray(participants) || participants.length === 0) {
        return res.status(400).json({
          error: 'At least one participant with a name and face photo is required.',
        });
      }

      const photos: Array<CachedParticipantPhoto | null> = [];
      for (let i = 0; i < participants.length; i += 1) {
        const pf = photoFiles[i];
        if (pf) {
          const buf = await fs.readFile(pf.path);
          photos.push({
            mimeType: pf.mimetype || 'image/jpeg',
            data: buf.toString('base64'),
          });
        } else {
          photos.push(null);
        }
      }

      const liveSessionId = `live-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      liveSessionCache.set(liveSessionId, {
        title,
        participants,
        photos,
        createdAt: Date.now(),
      });

      return res.json({ ok: true, liveSessionId });
    } catch (error) {
      return res
        .status(500)
        .json({ error: formatServerError(error, 'Failed to initialize live session.') });
    } finally {
      await Promise.all(
        tempPaths.map(async (p) => {
          try {
            await fs.unlink(p);
          } catch {
            // Ignore cleanup errors
          }
        })
      );
    }
  }
);

app.post(
  '/api/meetings/transcribe-live-chunk',
  upload.fields([
    { name: 'chunk', maxCount: 1 },
    { name: 'participantPhotos', maxCount: 24 },
  ]),
  async (req, res) => {
    const files = req.files as
      | { [fieldname: string]: Express.Multer.File[] }
      | undefined;
    const chunkFile = files?.chunk?.[0];
    const photoFiles = files?.participantPhotos ?? [];
    const tempPaths: string[] = [];
    if (chunkFile) tempPaths.push(chunkFile.path);
    for (const pf of photoFiles) tempPaths.push(pf.path);

    try {
      if (!chunkFile) {
        return res.status(400).json({ error: 'A live video/audio clip is required.' });
      }

      const liveSessionId = String(req.body?.liveSessionId ?? '').trim();
      const cached = liveSessionId ? liveSessionCache.get(liveSessionId) : undefined;

      let title = cached?.title ?? String(req.body?.title ?? 'Live Meeting').trim();
      let participants = cached?.participants ?? [];
      let photos: Array<CachedParticipantPhoto | null> = cached?.photos ?? [];

      if (!cached) {
        participants = JSON.parse(String(req.body?.participants ?? '[]'));
        if (!Array.isArray(participants) || participants.length === 0) {
          return res.status(400).json({
            error: 'Live session not found or missing participant metadata.',
          });
        }
        photos = [];
        for (let i = 0; i < participants.length; i += 1) {
          const pf = photoFiles[i];
          if (pf) {
            const buf = await fs.readFile(pf.path);
            photos.push({
              mimeType: pf.mimetype || 'image/jpeg',
              data: buf.toString('base64'),
            });
          } else {
            photos.push(null);
          }
        }
      }

      const offsetSec = Number(req.body?.offsetSec ?? 0);
      const ai = getGeminiClient();
      const parts: Array<Record<string, unknown>> = [];

      parts.push({
        text: [
          `You are transcribing a LIVE segment starting at ${offsetSec}s of the meeting "${title}".`,
          `Match any visible speaker's face against the registered participants below, classify their facial emotion while speaking, transcribe spoken speech verbatim, and highlight any task or commitment spoken in this segment.`,
          `Registered participants:`,
          ...participants.map(
            (p, idx) =>
              `${idx + 1}. Name: "${p.name}" | ID: "${p.id}"${p.role ? ` | Role: "${p.role}"` : ''}`
          ),
        ].join('\n'),
      });

      for (let i = 0; i < participants.length; i += 1) {
        const participant = participants[i];
        const photo = photos[i];
        if (photo) {
          parts.push({
            text: `Reference face photo for participant "${participant.name}" (speakerId: "${participant.id}"):`,
          });
          parts.push({
            inlineData: {
              mimeType: photo.mimeType,
              data: photo.data,
            },
          });
        }
      }

      const chunkBuffer = await fs.readFile(chunkFile.path);
      const chunkMime = chunkFile.mimetype || String(req.body?.chunkMimeType || 'video/webm');
      parts.push({
        inlineData: {
          mimeType: chunkMime,
          data: chunkBuffer.toString('base64'),
        },
      });

      parts.push({
        text: [
          `Live Segment Instructions:`,
          `1. Return startSec and endSec relative to the start of THIS clip (starting from 0).`,
          `2. If there is only silence, background noise, or no intelligible speech in this clip, return empty arrays: { "durationSec": 6, "transcript": [], "actionItems": [] }. Do NOT invent dialogue.`,
          `3. Match speaker faces to the registered reference photos using exact speakerId and speakerName (or "unknown" / "Unregistered Speaker" if not registered).`,
          `4. Classify facial emotion strictly as one of: "Confident", "Focused", "Enthusiastic", "Concerned", "Hesitant", "Frustrated", "Supportive", "Neutral", with facialCue and emotionIntensity (0-100).`,
          `5. Highlight any commitment or task in highlightedPhrase and add a matching item in actionItems.`,
        ].join('\n'),
      });

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts: parts as never },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: TRANSCRIPTION_RESPONSE_SCHEMA,
        },
      });

      const parsed = parseGeminiJsonObject(
        response.text,
        'transcribing the live video clip'
      );
      return res.json(normalizeTranscriptionPayload(parsed, participants));
    } catch (error) {
      return res.status(500).json({
        error: formatServerError(error, 'Failed to transcribe live video segment.'),
      });
    } finally {
      await Promise.all(
        tempPaths.map(async (p) => {
          try {
            await fs.unlink(p);
          } catch {
            // Ignore cleanup errors
          }
        })
      );
    }
  }
);

app.post('/api/meetings/live-session/end', (req, res) => {
  const liveSessionId = String(req.body?.liveSessionId ?? '').trim();
  if (liveSessionId) {
    liveSessionCache.delete(liveSessionId);
  }
  return res.json({ ok: true });
});

app.post(
  '/api/meetings/upload-chunk',
  upload.single('chunk'),
  async (req, res) => {
    const tempChunkPath = req.file?.path;
    try {
      const safeUploadId = String(req.body?.uploadId ?? '').replace(
        /[^a-zA-Z0-9_-]/g,
        ''
      );
      const chunkIndex = Number(req.body?.chunkIndex);
      const totalChunks = Number(req.body?.totalChunks);

      if (
        !safeUploadId ||
        !req.file ||
        !Number.isInteger(chunkIndex) ||
        !Number.isInteger(totalChunks) ||
        totalChunks < 1 ||
        chunkIndex < 0 ||
        chunkIndex >= totalChunks
      ) {
        return res.status(400).json({ error: 'Invalid video chunk payload.' });
      }

      const chunkDir = path.join(uploadDir, 'chunks', safeUploadId);
      await fs.mkdir(chunkDir, { recursive: true });
      const targetChunkPath = path.join(
        chunkDir,
        String(chunkIndex).padStart(6, '0')
      );
      await fs.copyFile(req.file.path, targetChunkPath);

      return res.json({ ok: true, chunkIndex });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to store video chunk.';
      return res.status(500).json({ error: message });
    } finally {
      if (tempChunkPath) {
        try {
          await fs.unlink(tempChunkPath);
        } catch {
          // Ignore cleanup errors
        }
      }
    }
  }
);

app.post(
  '/api/meetings/transcribe',
  upload.fields([
    { name: 'video', maxCount: 1 },
    { name: 'participantPhotos', maxCount: 24 },
  ]),
  async (req, res) => {
    const files = req.files as
      | { [fieldname: string]: Express.Multer.File[] }
      | undefined;
    const videoFile = files?.video?.[0];
    const photoFiles = files?.participantPhotos ?? [];
    const tempPathsToClean: string[] = [];
    let chunkDirToClean: string | null = null;

    if (videoFile) tempPathsToClean.push(videoFile.path);
    for (const pf of photoFiles) tempPathsToClean.push(pf.path);

    let uploadedGeminiFileName: string | null = null;

    try {
      await fs.mkdir(uploadDir, { recursive: true });
      let resolvedVideoPath = videoFile?.path ?? '';
      let resolvedVideoSize = videoFile?.size ?? 0;
      let resolvedVideoMime =
        videoFile?.mimetype || String(req.body?.videoMimeType || 'video/mp4');

      if (!videoFile) {
        const safeUploadId = String(req.body?.uploadId ?? '').replace(
          /[^a-zA-Z0-9_-]/g,
          ''
        );
        const totalChunks = Number(req.body?.totalChunks ?? 0);

        if (
          !safeUploadId ||
          !Number.isInteger(totalChunks) ||
          totalChunks < 1
        ) {
          return res.status(400).json({ error: 'A meeting video file is required.' });
        }

        const chunkDir = path.join(uploadDir, 'chunks', safeUploadId);
        chunkDirToClean = chunkDir;
        const assembledPath = path.join(
          uploadDir,
          `assembled-${safeUploadId}-${Date.now()}`
        );
        tempPathsToClean.push(assembledPath);

        const handle = await fs.open(assembledPath, 'w');
        try {
          for (let i = 0; i < totalChunks; i += 1) {
            const chunkPath = path.join(chunkDir, String(i).padStart(6, '0'));
            const chunkData = await fs.readFile(chunkPath);
            await handle.write(chunkData);
            resolvedVideoSize += chunkData.length;
          }
        } finally {
          await handle.close();
        }
        resolvedVideoPath = assembledPath;
      }

      const title = String(req.body?.title ?? 'Untitled Meeting').trim();
      const rawParticipants = String(req.body?.participants ?? '[]');
      const participants: Array<{ id: string; name: string; role?: string }> =
        JSON.parse(rawParticipants);

      if (!Array.isArray(participants) || participants.length === 0) {
        return res.status(400).json({
          error: 'At least one participant with a name and face photo is required.',
        });
      }

      const ai = getGeminiClient();

      // Build multimodal parts: participant face references first, then video
      const parts: Array<Record<string, unknown>> = [];

      parts.push({
        text: [
          `You are analyzing a meeting/reunion video titled "${title}".`,
          `Below are reference face photos for the ${participants.length} registered participants.`,
          `Use these faces to identify WHO is speaking at each moment in the video, analyze their FACIAL EMOTION while they speak, transcribe their speech verbatim, and highlight key action items / tasks automatically BEFORE any summary is generated.`,
          `Registered participants:`,
          ...participants.map(
            (p, idx) =>
              `${idx + 1}. Name: "${p.name}" | ID: "${p.id}"${p.role ? ` | Role: "${p.role}"` : ''}`
          ),
        ].join('\n'),
      });

      for (let i = 0; i < participants.length; i += 1) {
        const participant = participants[i];
        const photoFile = photoFiles[i];
        if (photoFile) {
          const photoBuffer = await fs.readFile(photoFile.path);
          parts.push({
            text: `Reference face photo for participant "${participant.name}" (speakerId: "${participant.id}"):`,
          });
          parts.push({
            inlineData: {
              mimeType: photoFile.mimetype || 'image/jpeg',
              data: photoBuffer.toString('base64'),
            },
          });
        }
      }

      if (resolvedVideoSize < 10 * 1024 * 1024) {
        const videoBuffer = await fs.readFile(resolvedVideoPath);
        parts.push({
          inlineData: {
            mimeType: resolvedVideoMime,
            data: videoBuffer.toString('base64'),
          },
        });
      } else {
        const uploaded = await ai.files.upload({
          file: resolvedVideoPath,
          config: { mimeType: resolvedVideoMime },
        });
        if (!uploaded.name) {
          throw new Error('Failed to upload video file to Gemini.');
        }
        uploadedGeminiFileName = uploaded.name;
        const activeFile = await waitForFileActive(ai, uploaded.name);
        parts.push({
          fileData: {
            fileUri: activeFile.uri,
            mimeType: activeFile.mimeType || resolvedVideoMime,
          },
        });
      }

      parts.push({
        text: [
          `Stage 1 Instructions (Transcription, Face-Speaker Matching, Facial Emotion & Action Highlighting):`,
          `1. Transcribe the meeting chronologically into speaker turns with accurate startSec and endSec timestamps.`,
          `2. Match each speaker's face in the video to one of the reference participant photos above and use their exact speakerId and speakerName. If a speaker is not in the reference list, use speakerId "unknown" and speakerName "Unregistered Speaker".`,
          `3. Observe the speaker's face during each turn and classify their facial emotion strictly as one of: "Confident", "Focused", "Enthusiastic", "Concerned", "Hesitant", "Frustrated", "Supportive", "Neutral". Also provide a short visual facialCue describing their facial expression (e.g. "furrowed brow, tight smile, leaning forward") and emotionIntensity (0-100).`,
          `4. Whenever a turn contains a commitment, task, follow-up, or decision assigned to someone, set highlightedPhrase to the exact verbatim substring inside text that states the action, AND create a corresponding item in actionItems linking sourceEntryId to that transcript entry's id and assigneeId/assigneeName to the responsible participant.`,
        ].join('\n'),
      });

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts: parts as never },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: TRANSCRIPTION_RESPONSE_SCHEMA,
        },
      });

      const parsed = parseGeminiJsonObject(
        response.text,
        'transcribing the meeting video'
      );
      return res.json(normalizeTranscriptionPayload(parsed, participants));
    } catch (error) {
      return res
        .status(500)
        .json({ error: formatServerError(error, 'Failed to transcribe meeting video.') });
    } finally {
      if (uploadedGeminiFileName) {
        try {
          const ai = getGeminiClient();
          await ai.files.delete({ name: uploadedGeminiFileName });
        } catch {
          // Ignore cleanup errors for ephemeral Gemini files
        }
      }
      if (chunkDirToClean) {
        try {
          await fs.rm(chunkDirToClean, { recursive: true, force: true });
        } catch {
          // Ignore cleanup errors
        }
      }
      await Promise.all(
        tempPathsToClean.map(async (filePath) => {
          try {
            await fs.unlink(filePath);
          } catch {
            // Ignore cleanup errors
          }
        })
      );
    }
  }
);

app.post('/api/meetings/summarize', async (req, res) => {
  try {
    const { title, participants, transcript, actionItems } = req.body ?? {};
    if (!Array.isArray(participants) || !Array.isArray(transcript)) {
      return res.status(400).json({
        error: 'Participants and transcript entries are required to generate the summary.',
      });
    }

    const ai = getGeminiClient();

    const prompt = [
      `Stage 2 Instructions (Generate Executive Summary & Per-Person Tasks/Emotions Summary):`,
      `Meeting Title: "${String(title ?? 'Meeting')}"`,
      `Participants: ${JSON.stringify(
        participants.map((p: { id: string; name: string; role?: string }) => ({
          id: p.id,
          name: p.name,
          role: p.role || '',
        }))
      )}`,
      `Transcribed Dialogue with Facial Emotions:\n${JSON.stringify(transcript)}`,
      `Highlighted Action Items & Tasks:\n${JSON.stringify(actionItems ?? [])}`,
      `Based ONLY on the verified transcript, facial emotion observations, and highlighted action items above:`,
      `1. Write a concise executiveSummary (3-4 sentences) and 2-4 concrete keyDecisions.`,
      `2. For EVERY registered participant, produce a personSummary containing:`,
      `   - participantId & participantName`,
      `   - dominantEmotion (one of Confident, Focused, Enthusiastic, Concerned, Hesitant, Frustrated, Supportive, Neutral)`,
      `   - emotionTrajectory (e.g. "Concerned -> Focused -> Supportive")`,
      `   - facialObservation (how their facial expressions evolved while speaking)`,
      `   - contributionSummary (what they advocated or raised in the meeting)`,
      `   - actionSummary (clear summary of all tasks/actions assigned to or taken on by this person)`,
    ].join('\n\n');

    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            executiveSummary: { type: Type.STRING },
            keyDecisions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            personSummaries: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  participantId: { type: Type.STRING },
                  participantName: { type: Type.STRING },
                  dominantEmotion: { type: Type.STRING },
                  emotionTrajectory: { type: Type.STRING },
                  facialObservation: { type: Type.STRING },
                  contributionSummary: { type: Type.STRING },
                  actionSummary: { type: Type.STRING },
                },
                required: [
                  'participantId',
                  'participantName',
                  'dominantEmotion',
                  'emotionTrajectory',
                  'facialObservation',
                  'contributionSummary',
                  'actionSummary',
                ],
              },
            },
          },
          required: ['executiveSummary', 'keyDecisions', 'personSummaries'],
        },
      },
    });

    const parsed = parseGeminiJsonObject(
      response.text,
      'generating the meeting summary'
    );
    return res.json(parsed);
  } catch (error) {
    return res
      .status(500)
      .json({ error: formatServerError(error, 'Failed to generate per-person summary.') });
  }
});

app.post('/api/meetings/chat', async (req, res) => {
  try {
    const { message, history, systemInstruction } = req.body ?? {};
    const trimmedMessage = String(message ?? '').trim();
    if (!trimmedMessage) {
      return res.status(400).json({ error: 'A question message is required.' });
    }

    const ai = getGeminiClient();

    const safeHistory = Array.isArray(history)
      ? history
          .filter(
            (item: { role?: string; parts?: Array<{ text?: string }> }) =>
              (item.role === 'user' || item.role === 'model') &&
              Array.isArray(item.parts) &&
              typeof item.parts[0]?.text === 'string' &&
              item.parts[0].text.trim().length > 0
          )
          .map((item: { role: 'user' | 'model'; parts: Array<{ text: string }> }) => ({
            role: item.role,
            parts: [{ text: item.parts[0].text }],
          }))
      : [];

    const chat = ai.chats.create({
      model: 'gemini-4-hydrogen',
      history: safeHistory,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        systemInstruction:
          typeof systemInstruction === 'string' && systemInstruction.trim()
            ? systemInstruction
            : 'You are a helpful meeting intelligence assistant answering questions strictly from the provided meeting transcripts, action items, and facial emotion observations.',
      },
    });

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(
        () => reject(new Error('Gemini chat request timed out. Please try again.')),
        25000
      );
    });

    const response = await Promise.race([
      chat.sendMessage({ message: trimmedMessage }),
      timeoutPromise,
    ]);
    return res.json({ reply: (response.text ?? '').trim() });
  } catch (error) {
    return res
      .status(500)
      .json({ error: formatServerError(error, 'Failed to answer meeting question.') });
  }
});

app.use('/api', (_req, res) => {
  return res.status(404).json({ error: 'API endpoint not found.' });
});

app.use(
  (
    err: unknown,
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) => {
    if (!req.path.startsWith('/api')) {
      return next(err);
    }
    if (err instanceof multer.MulterError) {
      const msg =
        err.code === 'LIMIT_FILE_SIZE'
          ? 'File exceeds the 250 MB maximum upload limit.'
          : `Upload error: ${err.message}`;
      return res.status(400).json({ error: msg });
    }
    return res
      .status(500)
      .json({ error: formatServerError(err, 'Unexpected server error.') });
  }
);

async function startServer() {
  await fs.mkdir(uploadDir, { recursive: true });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vitePromise = createViteServer({
      server: { middlewareMode: true, allowedHosts: true },
      appType: 'spa',
    });

    app.use(async (req, res, next) => {
      try {
        const vite = await vitePromise;
        vite.middlewares(req, res, next);
      } catch (err) {
        next(err);
      }
    });

    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Cadence server listening on http://localhost:${PORT}`);
    });

    await vitePromise;
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });

    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Cadence server listening on http://localhost:${PORT}`);
    });
  }
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled rejection in Cadence server:', reason);
});

startServer().catch((err) => {
  console.error('Failed to start Cadence server:', err);
  process.exit(1);
});
