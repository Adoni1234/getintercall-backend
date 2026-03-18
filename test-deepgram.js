// test-deepgram.js — corre con: node test-deepgram.js
const WebSocket = require('ws');

const API_KEY = '7c168ea9008d6f45d363ed381e9834bcd03b842e';

// TEST 1: mínimos
const params = new URLSearchParams({
  model: 'nova-2',
  encoding: 'linear16',
  sample_rate: '16000',
});

const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;
console.log('Conectando a:', url);

const ws = new WebSocket(url, {
  headers: { Authorization: `Token ${API_KEY}` },
});

ws.on('open', () => {
  console.log('✅ CONEXIÓN ABIERTA — la key funciona!');
  // Enviar silencio para ver si responde
  const silence = Buffer.alloc(3200, 0);
  ws.send(silence);
  setTimeout(() => {
    ws.send(JSON.stringify({ type: 'CloseStream' }));
    setTimeout(() => ws.close(), 500);
  }, 2000);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());
  console.log('📨 Mensaje:', msg.type, msg.channel?.alternatives?.[0]?.transcript || '');
});

ws.on('unexpected-response', (req, res) => {
  let body = '';
  res.on('data', c => body += c);
  res.on('end', () => {
    console.log('❌ HTTP', res.statusCode, ':', body);
  });
});

ws.on('error', (err) => {
  console.log('❌ Error:', err.message);
});

ws.on('close', (code) => {
  console.log('🔌 Cerrado con código:', code);
  process.exit(0);
});

setTimeout(() => {
  console.log('⏰ Timeout — sin respuesta en 5s');
  process.exit(1);
}, 5000);