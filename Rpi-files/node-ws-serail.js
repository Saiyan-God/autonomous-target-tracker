// Require the serialport node module
var SerialPort = require('serialport');
//var SerialPort = serialport.SerialPort;

const WebSocket = require('ws');
 // Open the port
// var port = new SerialPort("/dev/tty17", {
//     baudRate: 9600,
//     parser: new SerialPort.parsers.Readline("\n")
// });
 
const wss = new WebSocket.Server({ port: 8080 });
 
wss.on('connection', function connection(ws) {
  ws.on('message', function incoming(message) {
    console.log('received: %s', message);
  });
 
  ws.send('something');
});

// // Read the port data
// port.on("open", function () {
//     console.log('open');
//     port.on('data', function(data) {
//         console.log(data);
//     });
// });