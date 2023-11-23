<!-- index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Client</title>
    <link rel="stylesheet" href="jquery.gritter.css">
    <style type="text/css">
        video{
            position: fixed;
            right: 0;
            bottom: 0;
            width: 100%;
            height: 100%;
        }
    </style>
</head>
<body>
    <video id="video-feed" autoplay></video>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/3.1.3/socket.io.js"></script>
    <script src="jquery.gritter.min.js"></script>
    <script src="speak.js"></script>
    <script>
        const socket = io.connect('https://10.35.20.92:5000');
        // Access the webcam and start streaming
        const constraints = {
		  	video: {
			    width: {
			      	min: 1280,
			      	ideal: 1920,
			      	max: 2560,
			    },
			    height: {
			      	min: 720,
			      	ideal: 1080,
			      	max: 1440,
			    },
			    facingMode: "environment" // cam sau
		  	},
		}
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            const video = document.getElementById('video-feed');
            video.srcObject = stream;

            // Capture frames and send to the server
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            setInterval(() => {
                // Draw the current frame on the canvas
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Get the base64-encoded frame from the canvas
                const frame = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

                // Send the frame to the server
                socket.emit('video_frame', frame);
            }, 1000);  // Send a frame every 100 milliseconds (adjust as needed)
        })
        .catch((error) => {
            console.error('Error accessing webcam:', error);
        });

        // received from the server
        socket.on('video_frame', (message) => {
            $.gritter.add({
                title: 'BIỂN BÁO',
                text: message,
                time: 2000,
                class_name: 'bottom-right'
            });
            speak("Chú ý có biển báo "+message);
        });
    </script>
</body>
</html>
