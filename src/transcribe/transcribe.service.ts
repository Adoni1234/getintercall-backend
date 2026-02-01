import { Injectable, Logger } from '@nestjs/common';
import { AssemblyAI } from 'assemblyai';

@Injectable()
export class TranscribeService {
  private readonly logger = new Logger(TranscribeService.name);
  private assembly: AssemblyAI | null;
  private sessionData = new Map<
    string,
    {
      lastFullTranscript: string;
      lastSentLength: number;
      firstPartialReceived: boolean;
    }
  >();

  constructor() {
    const apiKey = process.env.ASSEMBLYAI_API_KEY;
    if (!apiKey) {
      this.logger.error('ASSEMBLYAI_API_KEY no encontrada – usando mock');
      this.assembly = null;
    } else {
      this.assembly = new AssemblyAI({ apiKey });
      this.logger.log('AssemblyAI v4 real-time listo con key');
    }
  }

  async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
    try {
      this.logger.log(
        `Batch: ${file.originalname}, tamaño: ${file.size} bytes`,
      );
      if (!this.assembly) {
        return { text: 'Mock batch: Audio procesado' };
      }
      const transcript = await this.assembly.transcripts.transcribe({
        audio: file.buffer,
        language_code: 'es',
      });

      if (transcript.status === 'completed') {
        this.logger.log(`Batch texto: ${transcript.text?.length || 0} chars`);
        return { text: transcript.text || '' };
      } else {
        this.logger.warn(`Batch status: ${transcript.status}`);
        return { text: '' };
      }
    } catch (error) {
      this.logger.error(`Batch error: ${error.message}`);
      return { text: '' };
    }
  }

  private detectLanguage(text: string): 'es' | 'en' {
    const cleanText = text.toLowerCase().trim();
    
    if (/[áéíóúñ¿¡]/i.test(cleanText)) {
      return 'es';
    }
    
    const spanishGrammarPatterns = [
      /\b(que|qué)\s+(es|son|está|están|tiene|tienen)\b/i,
      /\b(el|la|los|las)\s+\w+\s+(de|del)\b/i,
      /\b(esto|esta|este|eso|esa|ese)\s+(es|son)\b/i,
      /\b(muy|más|menos)\s+\w+/i,
      /\b(no|si)\s+(puedo|puede|quiero|quiere|voy|va)\b/i,
      /\baquí\s+(es|está|en)\b/i,
      /\bestamos\s+(con|en)\b/i,
    ];
    
    if (spanishGrammarPatterns.some(pattern => pattern.test(cleanText))) {
      return 'es';
    }
    
    const spanishPattern = /\b(de|del|el|la|los|las|un|una|está|están|son|es|como|qué|cómo|por|para|con|sin|pero|y|o|mi|tu|su|me|te|se|lo|le|ha|he|sido|sé|vamos|hacer|entonces|solo|mientras|lugares|más|nada|esto|no|que|muy|aquí|allí|allá|ahí|bien|mal|todo|siempre|nunca|cuando|donde|mucho|poco|grande|nuevo|bueno|malo|si|sí|ver|vea|veía|ir|voy|va|hacer|hago|dice|decir|ser|estar|tener|tengo|tiene|poder|puedo|puede|querer|quiero|deber|debe|año|día|vez|cosa|gente|tiempo|vida|casa|ciudad|centro|corazón|velada|desde|hasta|otro|mismo|cada|todos|sufro|huevo|viéndome|estamos|sea|medellín|raro|querer)\b/gi;
    
    const words = cleanText.split(/\s+/).filter(w => w.length > 0);
    const spanishMatches = cleanText.match(spanishPattern);
    const spanishWordCount = spanishMatches ? spanishMatches.length : 0;
    const spanishRatio = spanishWordCount / words.length;
    
    if (words.length <= 5 && spanishWordCount >= 1) {
      return 'es';
    }
    
    if (spanishRatio >= 0.18) {
      this.logger.log(`🎯 Spanish detected: ${(spanishRatio * 100).toFixed(1)}% (${spanishWordCount}/${words.length} words)`);
      return 'es';
    }
    
    return 'en';
  }

  async startRealTimeTranscription(
    sessionId: string,
    callback: (partialData: string) => void,
  ): Promise<any> {
    this.logger.log(`Iniciando v4 real-time para session ${sessionId}`);
    if (!this.assembly) {
      this.logger.log(`Mock real-time para ${sessionId}`);
      const mockInterval = setInterval(() => {
        const mockData = JSON.stringify({
          text: `Mock partial [${sessionId}]: Hablando en vivo...`,
          language: 'es',
          isNewTurn: false,
          sessionId: sessionId,
        });
        callback(mockData);
      }, 2000);
      return { send: () => {}, close: () => clearInterval(mockInterval) };
    }

    this.sessionData.set(sessionId, {
      lastFullTranscript: '',
      lastSentLength: 0,
      firstPartialReceived: false,
    });

    try {
      const config = {
        sampleRate: 16000,
        speechModel: 'universal-streaming-multilingual' as any,
        
        // 🔥 VAD COMPLETAMENTE DESACTIVADO
        vad_threshold: 0.0,
        
        // 🔥 TIMEOUTS MUY LARGOS - NUNCA cortar
        end_silence_timeout: 3.0,           // ← 3 segundos
        max_end_of_turn_silence_ms: 3000,
        
        disable_partial_transcripts: false,
        word_boost: [], 
        boost_param: 'default' as any,
      };

      this.logger.log(`🎤 Config ULTRA sensible: VAD=0.0, silence=3.0s, NO filtros`);

      const transcriber = this.assembly.streaming.transcriber(config);

      let isOpen = false;
      const audioBuffer = new Int16Array(1600); // 100ms buffer at 16kHz
      let bufferIndex = 0;

      (transcriber.on as any)('open', (data: any) => {
        isOpen = true;
        this.logger.log(`v4 WS abierto para ${sessionId} (ID: ${data.id})`);
        if (bufferIndex > 0) {
          const chunk = audioBuffer.slice(0, bufferIndex);
          const arrayBuffer = chunk.buffer.slice(
            chunk.byteOffset,
            chunk.byteOffset + chunk.byteLength,
          );
          transcriber.sendAudio(arrayBuffer);
          this.logger.log(`Buffered chunk enviado: ${bufferIndex} samples`);
          bufferIndex = 0;
        }
      });

      (transcriber.on as any)('turn', (data: any) => {
        const fullTranscript = data.transcript || '';
        const isFinal = data.is_final || false;

        if (!fullTranscript.trim()) {
          return;
        }

        const detectedLang = this.detectLanguage(fullTranscript);
        this.logger.log(`🌐 Detected [${sessionId}]: ${detectedLang} for "${fullTranscript.substring(0, 30)}..."`);

        const session = this.sessionData.get(sessionId);
        if (!session) {
          this.logger.warn(`Session ${sessionId} not found in turn event`);
          return;
        }

        let textToSend = '';

        if (isFinal) {
          textToSend = fullTranscript.trim();
          this.logger.log(
            `✅ FINAL [${sessionId}] [${detectedLang}]: "${textToSend.substring(0, 60)}..."`,
          );

          session.lastFullTranscript = '';
          session.lastSentLength = 0;
          session.firstPartialReceived = false;

          const partialData = JSON.stringify({
            text: textToSend,
            language: detectedLang,
            isNewTurn: true,
            sessionId: sessionId,
          });
          callback(partialData);
        } else {
          if (fullTranscript.length > session.lastSentLength) {
            textToSend = fullTranscript
              .substring(session.lastSentLength)
              .trim();

            session.lastFullTranscript = fullTranscript;
            session.lastSentLength = fullTranscript.length;

            if (textToSend) {
              this.logger.log(
                `📝 PARTIAL [${sessionId}] [${detectedLang}]: New="${textToSend.substring(0, 40)}..." (sent: ${session.lastSentLength}/${fullTranscript.length})`,
              );

              const partialData = JSON.stringify({
                text: textToSend,
                language: detectedLang,
                isNewTurn: false,
                sessionId: sessionId,
              });
              callback(partialData);
            }
          } else if (fullTranscript.length < session.lastSentLength) {
            this.logger.log(
              `🔄 REFORMULATION ignored [${sessionId}]: Old=${session.lastSentLength} → New=${fullTranscript.length}`,
            );
            session.lastFullTranscript = fullTranscript;
            session.lastSentLength = fullTranscript.length;
          } else {
            this.logger.log(
              `⏭️ DUPLICATE ignored [${sessionId}]: Same length ${fullTranscript.length}`,
            );
          }
        }
      });

      transcriber.on('error', (error: any) => {
        this.logger.error(`v4 error [${sessionId}]: ${error.message}`);
        const fallbackData = JSON.stringify({
          text: 'Fallback: Audio detectado (error en API)',
          language: 'en',
          isNewTurn: true,
          sessionId: sessionId,
        });
        callback(fallbackData);
      });

      transcriber.on('close', (code: number, reason: string) => {
        this.logger.log(
          `v4 WS cerrado para ${sessionId} (code: ${code}, reason: ${reason})`,
        );
        this.sessionData.delete(sessionId);
        isOpen = false;
      });

      await transcriber.connect();
      this.logger.log(`transcriber.connect() completado para ${sessionId}`);

      const sendChunk = (chunk: ArrayBuffer) => {
        const pcmData = new Int16Array(chunk);
        for (let i = 0; i < pcmData.length; i++) {
          audioBuffer[bufferIndex++] = pcmData[i];
          if (bufferIndex >= audioBuffer.length) {
            const sendBuffer = audioBuffer.slice(0, bufferIndex);
            const arrayBuffer = sendBuffer.buffer.slice(
              sendBuffer.byteOffset,
              sendBuffer.byteOffset + sendBuffer.byteLength,
            );
            if (isOpen) {
              transcriber.sendAudio(arrayBuffer);
              
              // 🔥 LOG NIVEL DE AUDIO para debug
              const avgLevel = this.calculateAudioLevel(new Int16Array(arrayBuffer));
              this.logger.log(
                `Chunk 100ms enviado a v4 para ${sessionId}: ${bufferIndex} samples, nivel: ${avgLevel.toFixed(2)}dB`,
              );
            } else {
              this.logger.log(`Chunk buffered hasta open para ${sessionId}`);
            }
            bufferIndex = 0;
          }
        }
      };

      return { send: sendChunk, close: () => transcriber.close() };
    } catch (error) {
      this.logger.error(`Error iniciando v4 [${sessionId}]: ${error.message}`);
      const fallbackInterval = setInterval(() => {
        const fallbackData = JSON.stringify({
          text: 'Fallback: Test transcripción en vivo...',
          language: 'en',
          isNewTurn: false,
          sessionId: sessionId,
        });
        callback(fallbackData);
      }, 2000);
      return { send: () => {}, close: () => clearInterval(fallbackInterval) };
    }
  }

  // 🔥 NUEVA FUNCIÓN: Calcular nivel de audio en backend
  private calculateAudioLevel(buffer: Int16Array): number {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      sum += Math.abs(buffer[i]);
    }
    const avg = sum / buffer.length;
    const normalized = avg / 32768;
    const db = 20 * Math.log10(normalized + 0.0001);
    return db;
  }
}
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';

// @Injectable()
// export class TranscribeService {
//   private readonly logger = new Logger(TranscribeService.name);
//   private assembly: AssemblyAI | null;
//   private sessionData = new Map<
//     string,
//     {
//       lastFullTranscript: string;
//       lastSentLength: number;
//       firstPartialReceived: boolean; // Track if we got first partial
//     }
//   >();

//   constructor() {
//     const apiKey = process.env.ASSEMBLYAI_API_KEY;
//     if (!apiKey) {
//       this.logger.error('ASSEMBLYAI_API_KEY no encontrada – usando mock');
//       this.assembly = null;
//     } else {
//       this.assembly = new AssemblyAI({ apiKey });
//       this.logger.log('AssemblyAI v4 real-time listo con key');
//     }
//   }

//   async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
//     try {
//       this.logger.log(
//         `Batch: ${file.originalname}, tamaño: ${file.size} bytes`,
//       );
//       if (!this.assembly) {
//         return { text: 'Mock batch: Audio procesado' };
//       }
//       const transcript = await this.assembly.transcripts.transcribe({
//         audio: file.buffer,
//         language_code: 'es',
//       });

//       if (transcript.status === 'completed') {
//         this.logger.log(`Batch texto: ${transcript.text?.length || 0} chars`);
//         return { text: transcript.text || '' };
//       } else {
//         this.logger.warn(`Batch status: ${transcript.status}`);
//         return { text: '' };
//       }
//     } catch (error) {
//       this.logger.error(`Batch error: ${error.message}`);
//       return { text: '' };
//     }
//   }

//   async startRealTimeTranscription(
//     sessionId: string,
//     callback: (partialData: string) => void,
//   ): Promise<any> {
//     this.logger.log(`Iniciando v4 real-time para session ${sessionId}`);
//     if (!this.assembly) {
//       this.logger.log(`Mock real-time para ${sessionId}`);
//       const mockInterval = setInterval(() => {
//         const mockData = JSON.stringify({
//           text: `Mock partial [${sessionId}]: Hablando en vivo...`,
//           lang: 'en',
//           isNewTurn: false,
//           sessionId: sessionId,
//         });
//         callback(mockData);
//       }, 2000);
//       return { send: () => {}, close: () => clearInterval(mockInterval) };
//     }

//     // Init session data - RESET on each new transcription
//     this.sessionData.set(sessionId, {
//       lastFullTranscript: '',
//       lastSentLength: 0,
//       firstPartialReceived: false,
//     });

//     try {
//       const config = {
//         sampleRate: 16000,
//         // Use multilingual model for English + Spanish support
//         speechModel: 'universal-streaming-multilingual' as any,
//         vad_threshold: 0.3, // Lower = more sensitive (catches quiet starts), 0.0-1.0
//         end_silence_timeout: 1.5, // 1.5s for balance
//         max_end_of_turn_silence_ms: 1500, // Match end_silence_timeout in ms
//         // Add these for better handling of pauses
//         disable_partial_transcripts: false, // Keep partials enabled
//         word_boost: [], // Can add specific words if needed
//         boost_param: 'default' as any,
//       };

//       this.logger.log(`Config v4: ${JSON.stringify(config)}`);

//       const transcriber = this.assembly.streaming.transcriber(config);

//       let isOpen = false;
//       const audioBuffer = new Int16Array(1600); // 100ms buffer at 16kHz
//       let bufferIndex = 0;

//       (transcriber.on as any)('open', (data: any) => {
//         isOpen = true;
//         this.logger.log(`v4 WS abierto para ${sessionId} (ID: ${data.id})`);
//         if (bufferIndex > 0) {
//           const chunk = audioBuffer.slice(0, bufferIndex);
//           const arrayBuffer = chunk.buffer.slice(
//             chunk.byteOffset,
//             chunk.byteOffset + chunk.byteLength,
//           );
//           transcriber.sendAudio(arrayBuffer);
//           this.logger.log(`Buffered chunk enviado: ${bufferIndex} samples`);
//           bufferIndex = 0;
//         }
//       });

//       (transcriber.on as any)('turn', (data: any) => {
//         const fullTranscript = data.transcript || '';
//         const isFinal = data.is_final || false;
//         const detectedLang = 'en';

//         if (!fullTranscript.trim()) {
//           return; // Ignore empty
//         }

//         const session = this.sessionData.get(sessionId);
//         if (!session) {
//           this.logger.warn(`Session ${sessionId} not found in turn event`);
//           return;
//         }

//         let textToSend = '';

//         if (isFinal) {
//           // FINAL: Send full text, then RESET for next turn
//           textToSend = fullTranscript.trim();
//           this.logger.log(
//             `✅ FINAL [${sessionId}]: "${textToSend.substring(0, 60)}..."`,
//           );

//           // Reset session for next turn
//           session.lastFullTranscript = '';
//           session.lastSentLength = 0;
//           session.firstPartialReceived = false; // Reset for next turn

//           const partialData = JSON.stringify({
//             text: textToSend,
//             lang: detectedLang,
//             isNewTurn: true,
//             sessionId: sessionId,
//           });
//           callback(partialData);
//         } else {
//           // PARTIAL: Extract only NEW text (diff from last sent)
//           if (fullTranscript.length > session.lastSentLength) {
//             // Extract new portion
//             textToSend = fullTranscript
//               .substring(session.lastSentLength)
//               .trim();

//             // Update tracking
//             session.lastFullTranscript = fullTranscript;
//             session.lastSentLength = fullTranscript.length;

//             if (textToSend) {
//               this.logger.log(
//                 `📝 PARTIAL [${sessionId}]: New="${textToSend.substring(0, 40)}..." (sent: ${session.lastSentLength}/${fullTranscript.length})`,
//               );

//               const partialData = JSON.stringify({
//                 text: textToSend,
//                 lang: detectedLang,
//                 isNewTurn: false,
//                 sessionId: sessionId,
//               });
//               callback(partialData);
//             }
//           } else if (fullTranscript.length < session.lastSentLength) {
//             // AssemblyAI reformulated (shorter) - likely a correction mid-stream
//             // Instead of sending reformulation, just reset tracking and wait for next partial
//             // This prevents jarring replacements in the UI
//             this.logger.log(
//               `🔄 REFORMULATION ignored [${sessionId}]: Old=${session.lastSentLength} → New=${fullTranscript.length}, waiting for continuation...`,
//             );
//             session.lastFullTranscript = fullTranscript;
//             session.lastSentLength = fullTranscript.length;
//             // Don't send anything - wait for next accumulation
//           } else {
//             // Same length - likely duplicate, ignore
//             this.logger.log(
//               `⏭️ DUPLICATE ignored [${sessionId}]: Same length ${fullTranscript.length}`,
//             );
//           }
//         }
//       });

//       transcriber.on('error', (error: any) => {
//         this.logger.error(`v4 error [${sessionId}]: ${error.message}`);
//         const fallbackData = JSON.stringify({
//           text: 'Fallback: Audio detectado (error en API)',
//           lang: 'en',
//           isNewTurn: true,
//           sessionId: sessionId,
//         });
//         callback(fallbackData);
//       });

//       transcriber.on('close', (code: number, reason: string) => {
//         this.logger.log(
//           `v4 WS cerrado para ${sessionId} (code: ${code}, reason: ${reason})`,
//         );
//         this.sessionData.delete(sessionId);
//         isOpen = false;
//       });

//       await transcriber.connect();
//       this.logger.log(`transcriber.connect() completado para ${sessionId}`);

//       const sendChunk = (chunk: ArrayBuffer) => {
//         const pcmData = new Int16Array(chunk);
//         for (let i = 0; i < pcmData.length; i++) {
//           audioBuffer[bufferIndex++] = pcmData[i];
//           if (bufferIndex >= audioBuffer.length) {
//             const sendBuffer = audioBuffer.slice(0, bufferIndex);
//             const arrayBuffer = sendBuffer.buffer.slice(
//               sendBuffer.byteOffset,
//               sendBuffer.byteOffset + sendBuffer.byteLength,
//             );
//             if (isOpen) {
//               transcriber.sendAudio(arrayBuffer);
//               this.logger.log(
//                 `Chunk 100ms enviado a v4 para ${sessionId}: ${bufferIndex} samples`,
//               );
//             } else {
//               this.logger.log(`Chunk buffered hasta open para ${sessionId}`);
//             }
//             bufferIndex = 0;
//           }
//         }
//       };

//       return { send: sendChunk, close: () => transcriber.close() };
//     } catch (error) {
//       this.logger.error(`Error iniciando v4 [${sessionId}]: ${error.message}`);
//       const fallbackInterval = setInterval(() => {
//         const fallbackData = JSON.stringify({
//           text: 'Fallback: Test transcripción en vivo...',
//           lang: 'en',
//           isNewTurn: false,
//           sessionId: sessionId,
//         });
//         callback(fallbackData);
//       }, 2000);
//       return { send: () => {}, close: () => clearInterval(fallbackInterval) };
//     }
//   }
// }
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';

// @Injectable()
// export class TranscribeService {
//   private readonly logger = new Logger(TranscribeService.name);
//   private assembly: AssemblyAI | null;

//   constructor() {
//     const apiKey = process.env.ASSEMBLYAI_API_KEY;
//     if (!apiKey) {
//       this.logger.error('ASSEMBLYAI_API_KEY no encontrada – usando mock');
//       this.assembly = null;
//     } else {
//       this.assembly = new AssemblyAI({ apiKey });
//       this.logger.log('AssemblyAI v4 real-time listo con key');
//     }
//   }

//   // Para batch (Postman)
//   async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
//     try {
//       this.logger.log(`Batch: ${file.originalname}, tamaño: ${file.size} bytes`);
//       if (!this.assembly) {
//         return { text: 'Mock batch: Audio procesado' };
//       }
//       const transcript = await this.assembly.transcripts.transcribe({
//         audio: file.buffer,
//         language_code: 'es',
//       });

//       if (transcript.status === 'completed') {
//         this.logger.log(`Batch texto: ${transcript.text?.length || 0} chars`);
//         return { text: transcript.text || '' };
//       } else {
//         this.logger.warn(`Batch status: ${transcript.status}`);
//         return { text: '' };
//       }
//     } catch (error) {
//       this.logger.error(`Batch error: ${error.message}`);
//       return { text: '' };
//     }
//   }

//   // Para real-time WS (multilingual auto-detect en/es)
//   async startRealTimeTranscription(sessionId: string, callback: (transcript: string) => void): Promise<any> {
//     this.logger.log(`Iniciando v4 real-time para session ${sessionId}`);
//     if (!this.assembly) {
//       // Mock
//       this.logger.log(`Mock real-time para ${sessionId}`);
//       const mockInterval = setInterval(() => {
//         const mockText = `Mock partial [${sessionId}]: Hablando en vivo... (tiempo ${Date.now() % 10000})`;
//         callback(mockText);
//       }, 2000);
//       return { send: () => {}, close: () => clearInterval(mockInterval) };
//     }

//     try {
//       const config = {
//         sampleRate: 16000,
//         speechModel: 'universal-streaming-multilingual' as const, // ← Fix: Literal string for type union
//         languageDetection: true, // ← Auto-detect idioma (en/es/fr/de/it/pt)
//         // Remueve language_code (auto-detect maneja)
//       };

//       this.logger.log(`Config v4 multilingual: ${JSON.stringify(config)}`);

//       const transcriber = this.assembly.streaming.transcriber(config);

//       let isOpen = false;
//       const audioBuffer = new Int16Array(800); // 50ms at 16kHz
//       let bufferIndex = 0;

//       transcriber.on('open', (data: any) => {
//         isOpen = true;
//         this.logger.log(`v4 WS abierto para ${sessionId} (ID: ${data.id})`);
//         // Send buffered if any
//         if (bufferIndex > 0) {
//           const chunk = audioBuffer.slice(0, bufferIndex);
//           const arrayBuffer = chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength);
//           transcriber.sendAudio(arrayBuffer);
//           this.logger.log(`Buffered chunk enviado: ${bufferIndex} samples`);
//           bufferIndex = 0;
//         }
//       });

//       transcriber.on('turn', (data: any) => {
//         const transcript = data.transcript;
//         const detectedLang = data.language_code || 'auto'; // Log idioma detectado
//         if (transcript) {
//           this.logger.log(`Partial v4 [${sessionId} - ${detectedLang}]: ${transcript}`);
//           callback(transcript);
//         }
//       });

//       transcriber.on('error', (error: any) => {
//         this.logger.error(`v4 error [${sessionId}]: ${error.message}`);
//         callback('Fallback partial: Audio detectado (error en API)');
//       });

//       transcriber.on('close', (code: number, reason: string) => {
//         this.logger.log(`v4 WS cerrado para ${sessionId} (code: ${code}, reason: ${reason})`);
//         isOpen = false;
//       });

//       await transcriber.connect();
//       this.logger.log(`transcriber.connect() completado para ${sessionId}`);

//       const sendChunk = (chunk: ArrayBuffer) => {
//         this.logger.log(`Debug sendChunk: chunk.byteLength = ${chunk.byteLength}`);
//         const pcmData = new Int16Array(chunk);
//         this.logger.log(`Debug pcmData.length = ${pcmData.length}, max vol = ${Math.max(...pcmData.map(Math.abs))}`);
//         // No VAD – envía todo
//         for (let i = 0; i < pcmData.length; i++) {
//           audioBuffer[bufferIndex++] = pcmData[i];
//           if (bufferIndex >= audioBuffer.length) {
//             // Send 50ms chunk
//             const sendBuffer = audioBuffer.slice(0, bufferIndex);
//             const arrayBuffer = sendBuffer.buffer.slice(sendBuffer.byteOffset, sendBuffer.byteOffset + sendBuffer.byteLength);
//             if (isOpen) {
//               transcriber.sendAudio(arrayBuffer);
//               this.logger.log(`Chunk 50ms enviado a v4 para ${sessionId}: ${bufferIndex} samples (vol: ${Math.max(...pcmData.map(Math.abs))})`);
//             } else {
//               this.logger.log(`Chunk buffered hasta open para ${sessionId}`);
//             }
//             bufferIndex = 0;
//           }
//         }
//         if (bufferIndex > 0) {
//           this.logger.log(`Partial buffer: ${bufferIndex} samples pendientes`);
//         }
//       };

//       return { send: sendChunk, close: () => transcriber.close() };
//     } catch (error) {
//       this.logger.error(`Error iniciando v4 [${sessionId}]: ${error.message}`);
//       const fallbackInterval = setInterval(() => callback('Fallback: Test transcripción en vivo...'), 2000);
//       return { send: () => {}, close: () => clearInterval(fallbackInterval) };
//     }
//   }
// }

// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';

// @Injectable()
// export class TranscribeService {
//   private readonly logger = new Logger(TranscribeService.name);
//   private assembly: AssemblyAI | null;

//   constructor() {
//     const apiKey = process.env.ASSEMBLYAI_API_KEY;
//     if (!apiKey) {
//       this.logger.error('ASSEMBLYAI_API_KEY no encontrada – usando mock');
//       this.assembly = null;
//     } else {
//       this.assembly = new AssemblyAI({ apiKey });
//       this.logger.log('AssemblyAI v4 real-time listo con key');
//     }
//   }

//   // Para batch (Postman)
//   async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
//     try {
//       this.logger.log(`Batch: ${file.originalname}, tamaño: ${file.size} bytes`);
//       if (!this.assembly) {
//         return { text: 'Mock batch: Audio procesado' };
//       }
//       const transcript = await this.assembly.transcripts.transcribe({
//         audio: file.buffer,
//         language_code: 'es',
//       });

//       if (transcript.status === 'completed') {
//         this.logger.log(`Batch texto: ${transcript.text?.length || 0} chars`);
//         return { text: transcript.text || '' };
//       } else {
//         this.logger.warn(`Batch status: ${transcript.status}`);
//         return { text: '' };
//       }
//     } catch (error) {
//       this.logger.error(`Batch error: ${error.message}`);
//       return { text: '' };
//     }
//   }

//   // Para real-time WS (no VAD, envía todo, logs debug)
//   async startRealTimeTranscription(sessionId: string, callback: (transcript: string) => void): Promise<any> {
//     this.logger.log(`Iniciando v4 real-time para session ${sessionId}`);
//     if (!this.assembly) {
//       // Mock
//       this.logger.log(`Mock real-time para ${sessionId}`);
//       const mockInterval = setInterval(() => {
//         const mockText = `Mock partial [${sessionId}]: Hablando en vivo... (tiempo ${Date.now() % 10000})`;
//         callback(mockText);
//       }, 2000);
//       return { send: () => {}, close: () => clearInterval(mockInterval) };
//     }

//     try {
//       const config = {
//         sampleRate: 16000,
//         language_code: 'en', // Test English, cambia a 'es' después
//       };

//       this.logger.log(`Config v4: ${JSON.stringify(config)}`);

//       const transcriber = this.assembly.streaming.transcriber(config);

//       let isOpen = false;
//       const audioBuffer = new Int16Array(800); // 50ms at 16kHz
//       let bufferIndex = 0;

//       transcriber.on('open', (data: any) => {
//         isOpen = true;
//         this.logger.log(`v4 WS abierto para ${sessionId} (ID: ${data.id})`);
//         // Send buffered if any
//         if (bufferIndex > 0) {
//           const chunk = audioBuffer.slice(0, bufferIndex);
//           const arrayBuffer = chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength);
//           transcriber.sendAudio(arrayBuffer);
//           this.logger.log(`Buffered chunk enviado: ${bufferIndex} samples`);
//           bufferIndex = 0;
//         }
//       });

//       transcriber.on('turn', (data: any) => {
//         const transcript = data.transcript;
//         if (transcript) {
//           this.logger.log(`Partial v4 [${sessionId}]: ${transcript}`);
//           callback(transcript);
//         }
//       });

//       transcriber.on('error', (error: any) => {
//         this.logger.error(`v4 error [${sessionId}]: ${error.message}`);
//         callback('Fallback partial: Audio detectado (error en API)');
//       });

//       transcriber.on('close', (code: number, reason: string) => {
//         this.logger.log(`v4 WS cerrado para ${sessionId} (code: ${code}, reason: ${reason})`);
//         isOpen = false;
//       });

//       await transcriber.connect();
//       this.logger.log(`transcriber.connect() completado para ${sessionId}`);

//       const sendChunk = (chunk: ArrayBuffer) => {
//         this.logger.log(`Debug sendChunk: chunk.byteLength = ${chunk.byteLength}`);
//         const pcmData = new Int16Array(chunk);
//         this.logger.log(`Debug pcmData.length = ${pcmData.length}, max vol = ${Math.max(...pcmData.map(Math.abs))}`);
//         // No VAD – envía todo
//         for (let i = 0; i < pcmData.length; i++) {
//           audioBuffer[bufferIndex++] = pcmData[i];
//           if (bufferIndex >= audioBuffer.length) {
//             // Send 50ms chunk
//             const sendBuffer = audioBuffer.slice(0, bufferIndex);
//             const arrayBuffer = sendBuffer.buffer.slice(sendBuffer.byteOffset, sendBuffer.byteOffset + sendBuffer.byteLength);
//             if (isOpen) {
//               transcriber.sendAudio(arrayBuffer);
//               this.logger.log(`Chunk 50ms enviado a v4 para ${sessionId}: ${bufferIndex} samples (vol: ${Math.max(...pcmData.map(Math.abs))})`);
//             } else {
//               this.logger.log(`Chunk buffered hasta open para ${sessionId}`);
//             }
//             bufferIndex = 0;
//           }
//         }
//         if (bufferIndex > 0) {
//           this.logger.log(`Partial buffer: ${bufferIndex} samples pendientes`);
//         }
//       };

//       return { send: sendChunk, close: () => transcriber.close() };
//     } catch (error) {
//       this.logger.error(`Error iniciando v4 [${sessionId}]: ${error.message}`);
//       const fallbackInterval = setInterval(() => callback('Fallback: Test transcripción en vivo...'), 2000);
//       return { send: () => {}, close: () => clearInterval(fallbackInterval) };
//     }
//   }
// }
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';

// @Injectable()
// export class TranscribeService {
//   private readonly logger = new Logger(TranscribeService.name);
//   private assembly: AssemblyAI | null; // ← Fix: | null para mock

//   constructor() {
//     const apiKey = process.env.ASSEMBLYAI_API_KEY;
//     if (!apiKey) {
//       this.logger.error(
//         'ASSEMBLYAI_API_KEY no encontrada – usando mock para test',
//       );
//       this.assembly = null; // ← Ahora OK con | null
//     } else {
//       this.assembly = new AssemblyAI({ apiKey });
//       this.logger.log('AssemblyAI real-time listo con key');
//     }
//   }

//   // Para batch (Postman)
//   async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
//     try {
//       this.logger.log(
//         `Batch: ${file.originalname}, tamaño: ${file.size} bytes`,
//       );
//       if (!this.assembly) {
//         return { text: 'Mock batch: Audio procesado' };
//       }
//       const transcript = await this.assembly.transcripts.transcribe({
//         audio: file.buffer,
//         language_code: 'es',
//       });

//       if (transcript.status === 'completed') {
//         this.logger.log(`Batch texto: ${transcript.text?.length || 0} chars`);
//         return { text: transcript.text || '' };
//       } else {
//         this.logger.warn(`Batch status: ${transcript.status}`);
//         return { text: '' };
//       }
//     } catch (error) {
//       this.logger.error(`Batch error: ${error.message}`);
//       return { text: '' };
//     }
//   }

//   // Para real-time WS
//   async startRealTimeTranscription(
//     sessionId: string,
//     callback: (transcript: string) => void,
//   ): Promise<any> {
//     this.logger.log(`Iniciando real-time para session ${sessionId}`);
//     if (!this.assembly) {
//       // Mock para test (emite partials cada 2s)
//       this.logger.log(`Mock real-time para ${sessionId}`);
//       const mockInterval = setInterval(() => {
//         const mockText = `Mock partial [${sessionId}]: Hablando en vivo... (tiempo ${Date.now() % 10000})`;
//         callback(mockText);
//       }, 2000);
//       return { close: () => clearInterval(mockInterval) };
//     }

//     try {
//       const config = {
//         sampleRate: 16000,
//         languageCode: 'es',
//       };

//       this.logger.log(`Config real-time: ${JSON.stringify(config)}`);

//       const ws = this.assembly.realtime.transcriber(config);

//       ws.on('open', () => {
//         this.logger.log(`AssemblyAI WS abierto para ${sessionId}`);
//       });

//       ws.on('transcript', (transcript) => {
//         if (transcript.text) {
//           this.logger.log(`Partial real [${sessionId}]: ${transcript.text}`);
//           callback(transcript.text);
//         }
//       });

//       ws.on('error', (error) => {
//         this.logger.error(`AssemblyAI error [${sessionId}]: ${error.message}`);
//         // Fallback mock si error
//         callback('Fallback partial: Audio detectado (error en API)');
//       });

//       ws.on('close', () => {
//         this.logger.log(`AssemblyAI WS cerrado para ${sessionId}`);
//       });

//       return ws;
//     } catch (error) {
//       this.logger.error(
//         `Error iniciando real-time [${sessionId}]: ${error.message}`,
//       );
//       // Fallback mock
//       const fallbackInterval = setInterval(() => {
//         callback('Fallback: Test transcripción en vivo...');
//       }, 2000);
//       return { close: () => clearInterval(fallbackInterval) };
//     }
//   }
// }

// import { Injectable, Logger } from '@nestjs/common';

// @Injectable()
// export class TranscribeService {
//   private readonly logger = new Logger(TranscribeService.name);
//   private readonly apiKey: string; // ← Tipado como string
//   private readonly uploadUrl = 'https://api.assemblyai.com/v2/upload';
//   private readonly transcriptUrl = 'https://api.assemblyai.com/v2/transcript';

//   constructor() {
//     const key = process.env.ASSEMBLYAI_API_KEY;
//     if (!key) {
//       throw new Error('ASSEMBLYAI_API_KEY environment variable is required');
//     }
//     this.apiKey = key; // ← Asigna a string
//   }

//   async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
//     try {
//       this.logger.log(`Procesando archivo: ${file.originalname}, tamaño: ${file.size} bytes`);

//       // 1. Determina Content-Type basado en extensión/MIME
//       const contentType = this.getContentType(file.originalname, file.mimetype);
//       this.logger.log(`Content-Type: ${contentType}`);

//       // 2. Sube buffer como binary a /v2/upload
//       const uploadUrl = await this.uploadAudio(file.buffer, contentType);
//       this.logger.log(`Upload URL obtenida: ${uploadUrl}`);

//       // 3. Crea transcript con la URL
//       const transcript = await this.createTranscript(uploadUrl);
//       this.logger.log(`Transcripción obtenida: ${transcript ? transcript.length : 0} chars`);

//       return { text: transcript || '' };

//     } catch (error) {
//       this.logger.error(`Error en transcripción: ${error.message}`, error.stack);
//       return { text: '' }; // Vacío para no romper live stream
//     }
//   }

//   private getContentType(originalname: string, mimetype: string): string {
//     const extension = originalname.split('.').pop()?.toLowerCase() || '';
//     if (extension === 'webm') return 'audio/webm';
//     if (extension === 'ogg' || extension === 'opus') return 'audio/ogg; codecs=opus';
//     if (mimetype.includes('webm')) return 'audio/webm';
//     if (mimetype.includes('ogg')) return 'audio/ogg';
//     return 'audio/webm'; // Default
//   }

//   private async uploadAudio(buffer: Buffer, contentType: string): Promise<string> {
//     const response = await fetch(this.uploadUrl, {
//       method: 'POST',
//       headers: {
//         authorization: this.apiKey,
//         'content-type': contentType,
//       },
//       body: buffer as BodyInit, // ← Cast para TS: Buffer como BodyInit
//     });

//     if (!response.ok) {
//       const errorText = await response.text();
//       throw new Error(`Upload error: ${response.status} - ${errorText}`);
//     }

//     const data = await response.json();
//     return data.upload_url; // ← URL pública del archivo
//   }

//   private async createTranscript(audioUrl: string): Promise<string> {
//     // Paso 1: POST para crear job
//     const response = await fetch(this.transcriptUrl, {
//       method: 'POST',
//       headers: {
//         authorization: this.apiKey,
//         'content-type': 'application/json',
//       },
//       body: JSON.stringify({
//         audio_url: audioUrl, // ← La URL del upload
//         language_code: 'es', // Español para YouTube
//       }),
//     });

//     if (!response.ok) {
//       const errorText = await response.text();
//       throw new Error(`Transcript create error: ${response.status} - ${errorText}`);
//     }

//     const { id } = await response.json();
//     this.logger.log(`Job creado: ${id}`);

//     // Paso 2: Poll hasta completed
//     let attempts = 0;
//     const maxAttempts = 10; // ~30s
//     while (attempts < maxAttempts) {
//       await new Promise(resolve => setTimeout(resolve, 3000));

//       const pollResponse = await fetch(`${this.transcriptUrl}/${id}`, {
//         headers: { authorization: this.apiKey },
//       });

//       if (!pollResponse.ok) {
//         throw new Error(`Poll error: ${pollResponse.status}`);
//       }

//       const data = await pollResponse.json();
//       this.logger.log(`Poll ${attempts + 1}: status = ${data.status}`);

//       if (data.status === 'completed') {
//         return data.text || '';
//       } else if (data.status === 'error') {
//         this.logger.warn(`AssemblyAI error: ${data.error || 'unknown'}`);
//         return ''; // Vacío para no speech/too short
//       }

//       attempts++;
//     }

//     throw new Error('Timeout: Job no completó');
//   }
// }
