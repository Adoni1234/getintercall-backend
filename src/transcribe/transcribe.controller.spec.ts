import { Test, TestingModule } from '@nestjs/testing';
import { TranscribeController } from './transcribe.controller';

describe('TranscribeController', () => {
  let controller: TranscribeController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [TranscribeController],
    }).compile();

    controller = module.get<TranscribeController>(TranscribeController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
