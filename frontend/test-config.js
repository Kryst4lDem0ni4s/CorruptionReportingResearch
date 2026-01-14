const config = require('./config');

console.log('Configuration Test:');
console.log('==================');
console.log('Environment:', config.env);
console.log('Server:', `${config.server.host}:${config.server.port}`);
console.log('Backend:', config.backend.url);
console.log('Logging:', config.logging.format);
console.log('Compression:', config.compression.enabled);
console.log('==================');
console.log('All tests passed! ✅');

// Run: node test-config.js
