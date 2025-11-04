setInterval(() => {
    fetch('/detection_status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('predictionLabel').textContent = data.label;
            document.getElementById('yawnCount').textContent = data.yawn_count;
            document.getElementById('eyeState').textContent = data.eye_state;
        });
}, 1000);
