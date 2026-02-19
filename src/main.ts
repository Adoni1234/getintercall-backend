import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.enableCors({
    origin: [
      'https://getintercall.vercel.app',
      'https://localhost:4200',
      'http://localhost:4200',
    ],
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
  });

  const port = process.env.PORT || 3000;
  await app.listen(port, '0.0.0.0');
  console.log(`Backend corriendo en puerto ${port}`);
}
bootstrap();
// import { NestFactory } from '@nestjs/core';
// import { AppModule } from './app.module';
// import { IoAdapter } from '@nestjs/platform-socket.io'; // ← Registra /socket.io
// import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';

// async function bootstrap() {
//   const app = await NestFactory.create(AppModule);

//   // Adapter para Socket.io (crea /socket.io route)
//   app.useWebSocketAdapter(new IoAdapter(app));

//   // CORS global (incluye Socket.io polling)
//   app.enableCors({
//     origin: 'http://localhost:4200',
//     methods: 'GET,POST,PUT,DELETE,OPTIONS', // ← OPTIONS para polling
//     allowedHeaders: 'Content-Type, Authorization',
//     credentials: true,
//     preflightContinue: false,
//   });

//   // Swagger (opcional)
//   const config = new DocumentBuilder()
//     .setTitle('GetIntercall API')
//     .setDescription('API para transcripción real-time')
//     .setVersion('1.0')
//     .addTag('Transcribe')
//     .build();
//   const document = SwaggerModule.createDocument(app, config);
//   SwaggerModule.setup('docs', app, document);

//   await app.listen(3000);
//   console.log('Backend corriendo en http://localhost:3000');
// }
// bootstrap();
