#!/usr/bin/env python3
import gi, cv2, numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

# Your camera's RTSP URL
rtsp_url = "rtsp://192.168.153.1:8899/stream1"

# Build the pipeline
pipeline = Gst.parse_launch(
    'udpsrc port=5000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" ! \
    rtpjitterbuffer latency=200 ! \
    rtph264depay ! \
    decodebin ! \
    videoconvert ! video/x-raw,format=BGR ! \
    queue leaky=downstream max-size-buffers=1 ! \
    appsink name=sink emit-signals=true sync=false drop=true'
)

# Retrieve the appsink element
sink = pipeline.get_by_name("sink")

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Pull samples and display
while True:
    sample = sink.emit("try-pull-sample", 100000000)  # 100 ms timeout
    if not sample:
        continue
    buf = sample.get_buffer()
    caps = sample.get_caps()
    w = caps.get_structure(0).get_value("width")
    h = caps.get_structure(0).get_value("height")
    success, info = buf.map(Gst.MapFlags.READ)
    if not success:
        continue
    frame = np.frombuffer(info.data, np.uint8).reshape((h, w, 3))
    buf.unmap(info)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()