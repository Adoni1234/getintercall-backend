import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { MulterModule } from '@nestjs/platform-express';
import { TranscribeModule } from './transcribe/transcribe.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    MulterModule.register({
      dest: './uploads',
    }),
    TranscribeModule,
  ],
})
export class AppModule {}
