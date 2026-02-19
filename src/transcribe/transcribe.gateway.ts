import {
  WebSocketGateway,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Logger } from '@nestjs/common';
import { TranscribeService } from './transcribe.service';

@WebSocketGateway({
  cors: {
    origin: [
      'http://localhost:4200',
      'https://localhost:4200',
      'getintercall-git-main-adonis-projects-9faf0f78.vercel.app',
      'https://getintercall.vercel.app',
    ],
    methods: ['GET', 'POST'],
    credentials: true,
  },
  transports: ['websocket', 'polling'],
})
export class TranscribeGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer() server: Server;
  private readonly logger = new Logger(TranscribeGateway.name);

  constructor(private transcribeService: TranscribeService) {}

  handleConnection(client: Socket) {
    // Inicializar queue vacía al conectar — lista para recibir chunks
    client.data.chunkQueue = [];
    client.data.ws = null;
    client.data.chunkCount = 0;
    this.logger.log(`Cliente conectado: ${client.id}`);
  }

  @SubscribeMessage('startTranscription')
  async handleStart(
    @MessageBody() data: { sessionId: string },
    @ConnectedSocket() client: Socket,
  ) {
    this.logger.log(
      `Recibido startTranscription para session ${data.sessionId} desde ${client.id}`,
    );

    // Asegurarse de que la queue existe ANTES de conectar AssemblyAI
    // Los chunks que lleguen mientras conecta se guardan aquí
    if (!client.data.chunkQueue) client.data.chunkQueue = [];

    try {
      this.logger.log(`Conectando AssemblyAI para ${data.sessionId}...`);

      const wsProxy = await this.transcribeService.startRealTimeTranscription(
        data.sessionId,
        (partialData: string) => {
          this.logger.log(
            `Emisión partial [${data.sessionId}]: ${partialData.substring(0, 50)}...`,
          );
          client.emit('partialTranscript', partialData);
        },
      );

      // AssemblyAI ya está listo — asignar proxy
      client.data.ws = wsProxy;

      // Flush de chunks acumulados durante la conexión
      const queue: ArrayBuffer[] = client.data.chunkQueue || [];
      if (queue.length > 0) {
        this.logger.log(
          `🔄 Flushing ${queue.length} chunks en cola para ${data.sessionId}`,
        );
        for (const queuedChunk of queue) {
          wsProxy.send(queuedChunk);
        }
        this.logger.log(`✅ ${queue.length} chunks enviados a AssemblyAI`);
      }

      // Limpiar queue — de aquí en adelante los chunks van directo
      client.data.chunkQueue = [];

      this.logger.log(`✅ AssemblyAI listo para ${data.sessionId}`);
      client.emit('started', {
        sessionId: data.sessionId,
        message: 'Real-time iniciado',
      });
    } catch (error) {
      this.logger.error(
        `❌ Error iniciando AssemblyAI para ${data.sessionId}: ${error.message}`,
      );
      client.emit('error', { message: error.message });
    }
  }

  @SubscribeMessage('audioChunk')
  handleAudioChunk(
    @MessageBody() data: { sessionId: string; chunk: number[] },
    @ConnectedSocket() client: Socket,
  ) {
    const chunkBuffer = new Uint8Array(data.chunk).buffer;
    const wsProxy = client.data.ws;

    if (wsProxy && wsProxy.send) {
      // AssemblyAI listo — enviar directo
      wsProxy.send(chunkBuffer);

      client.data.chunkCount = (client.data.chunkCount || 0) + 1;
      if (client.data.chunkCount % 40 === 0) {
        this.logger.log(
          `📤 Chunk #${client.data.chunkCount} enviado para ${data.sessionId}: ${chunkBuffer.byteLength} bytes`,
        );
      }
    } else {
      // AssemblyAI aún conectando — guardar en queue
      if (!client.data.chunkQueue) client.data.chunkQueue = [];
      client.data.chunkQueue.push(chunkBuffer);

      // Log solo cada 10 chunks para no saturar
      if (client.data.chunkQueue.length % 10 === 0) {
        this.logger.warn(
          `⏳ ${client.data.chunkQueue.length} chunks en cola para ${data.sessionId} (AssemblyAI conectando...)`,
        );
      }
    }
  }

  @SubscribeMessage('stopTranscription')
  handleStop(
    @MessageBody() data: { sessionId: string },
    @ConnectedSocket() client: Socket,
  ) {
    this.logger.log(
      `Recibido stop para session ${data.sessionId} desde ${client.id}`,
    );
    const wsProxy = client.data.ws;
    if (wsProxy && wsProxy.close) {
      wsProxy.close();
      client.data.ws = null;
      client.data.chunkQueue = [];
      client.data.chunkCount = 0;
      this.logger.log(`🛑 WS AssemblyAI cerrado para ${data.sessionId}`);
    }
    client.emit('stopped', { sessionId: data.sessionId });
  }

  handleDisconnect(client: Socket) {
    this.logger.log(`Cliente desconectado: ${client.id}`);
    const wsProxy = client.data.ws;
    if (wsProxy && wsProxy.close) {
      wsProxy.close();
      client.data.ws = null;
      client.data.chunkQueue = [];
    }
  }
}

// import {
//   WebSocketGateway,
//   SubscribeMessage,
//   MessageBody,
//   ConnectedSocket,
//   WebSocketServer,
//   OnGatewayConnection,
//   OnGatewayDisconnect,
// } from '@nestjs/websockets';
// import { Server, Socket } from 'socket.io';
// import { Logger } from '@nestjs/common';
// import { TranscribeService } from './transcribe.service';

// @WebSocketGateway({
//   cors: {
//     origin: [
//       'http://localhost:4200',
//       'https://localhost:4200',
//       'getintercall-git-main-adonis-projects-9faf0f78.vercel.app',
//       'https://getintercall.vercel.app',
//     ],
//     methods: ['GET', 'POST'],
//     credentials: true,
//   },
// })
// export class TranscribeGateway
//   implements OnGatewayConnection, OnGatewayDisconnect
// {
//   @WebSocketServer() server: Server;
//   private readonly logger = new Logger(TranscribeGateway.name);

//   constructor(private transcribeService: TranscribeService) {}

//   handleConnection(client: Socket) {
//     this.logger.log(`Cliente conectado: ${client.id}`);
//   }

//   @SubscribeMessage('startTranscription')
//   async handleStart(
//     @MessageBody() data: { sessionId: string },
//     @ConnectedSocket() client: Socket,
//   ) {
//     this.logger.log(
//       `Recibido startTranscription para session ${data.sessionId} desde ${client.id}`,
//     );
//     try {
//       this.logger.log(`Llamando service para ${data.sessionId}`);
//       const wsProxy = await this.transcribeService.startRealTimeTranscription(
//         data.sessionId,
//         (partialData: string) => {
//           this.logger.log(
//             `Emisión partial desde Gateway [${data.sessionId}]: ${partialData.substring(0, 50)}...`,
//           );
//           client.emit('partialTranscript', partialData); // String JSON
//         },
//       );
//       client.data.ws = wsProxy;
//       client.data.chunkQueue = []; // Queue for delayed chunks
//       this.logger.log(`WS AssemblyAI asignado para ${data.sessionId}`);
//       client.emit('started', {
//         sessionId: data.sessionId,
//         message: 'Real-time iniciado',
//       }); // ← Emit after await
//       // Flush queued chunks
//       const queue = client.data.chunkQueue || [];
//       for (const queuedChunk of queue) {
//         wsProxy.send(queuedChunk);
//         this.logger.log(
//           `Queued chunk flushed for ${data.sessionId}: ${queuedChunk.byteLength} bytes`,
//         );
//       }
//       delete client.data.chunkQueue;
//     } catch (error) {
//       this.logger.error(
//         `Error iniciando AssemblyAI para ${data.sessionId}: ${error.message}`,
//       );
//       client.emit('error', { message: error.message });
//     }
//   }

//   @SubscribeMessage('audioChunk')
//   handleAudioChunk(
//     @MessageBody() data: { sessionId: string; chunk: number[] },
//     @ConnectedSocket() client: Socket,
//   ) {
//     const chunkArray = data.chunk;
//     const chunkBuffer = new Uint8Array(chunkArray).buffer;
//     this.logger.log(
//       `Recibido chunk para session ${data.sessionId}: ${chunkBuffer.byteLength} bytes desde ${client.id}`,
//     );
//     const wsProxy = client.data.ws;
//     if (wsProxy && wsProxy.send) {
//       wsProxy.send(chunkBuffer);
//       this.logger.log(
//         `Chunk enviado via proxy para ${data.sessionId}, reconstructed ${chunkBuffer.byteLength} bytes`,
//       );
//     } else {
//       // Queue if not ready
//       if (!client.data.chunkQueue) client.data.chunkQueue = [];
//       client.data.chunkQueue.push(chunkBuffer);
//       this.logger.warn(
//         `Chunk queued for session ${data.sessionId} (proxy not ready yet)`,
//       );
//     }
//     client.emit('chunkReceived', {
//       sessionId: data.sessionId,
//       size: chunkBuffer.byteLength,
//     });
//   }

//   @SubscribeMessage('stopTranscription')
//   handleStop(
//     @MessageBody() data: { sessionId: string },
//     @ConnectedSocket() client: Socket,
//   ) {
//     this.logger.log(
//       `Recibido stop para session ${data.sessionId} desde ${client.id}`,
//     );
//     const wsProxy = client.data.ws;
//     if (wsProxy && wsProxy.close) {
//       wsProxy.close();
//       client.data.ws = null;
//       delete client.data.chunkQueue;
//       this.logger.log(`WS AssemblyAI cerrado para ${data.sessionId}`);
//     }
//     client.emit('stopped', { sessionId: data.sessionId });
//   }

//   handleDisconnect(client: Socket) {
//     this.logger.log(`Cliente desconectado: ${client.id}`);
//     const wsProxy = client.data.ws;
//     if (wsProxy && wsProxy.close) {
//       wsProxy.close();
//       client.data.ws = null;
//       delete client.data.chunkQueue;
//     }
//   }
// }
