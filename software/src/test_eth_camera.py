import gi, cv2, numpy as np
import time
import re # Import regex module
import sys


HEADLESS = True  # set to True to record video instead of display
DURATION = 5  # seconds to record when headless
DEFAULT_FPS = 30 # Default FPS if not found in caps
MAX_WAIT_TIME = 10  # Maximum seconds to wait for first packet

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# 1. Initialize GStreamer
Gst.init(None)

# Print some debug info
print(f"Debug: Running on Python {sys.version}")
print(f"Debug: OpenCV version: {cv2.__version__}")
print(f"Debug: GStreamer version: {Gst.version_string()}")

# 2. First, build a minimal pipeline to verify UDP reception before proceeding
print("Setting up test UDP pipeline to verify data reception...")
test_pipeline = Gst.parse_launch(
    'udpsrc port=5000 ! fakesink name=testfakesink sync=false'
)

# Start test pipeline
print("Starting test UDP pipeline...")
ret = test_pipeline.set_state(Gst.State.PLAYING)
print(f"Test pipeline state change result: {ret}")

# Test for UDP packet reception
start_time = time.time()
print(f"Waiting up to {MAX_WAIT_TIME} seconds for UDP packet...")
packet_received = False

# Get bus for test pipeline
test_bus = test_pipeline.get_bus()

while time.time() - start_time < MAX_WAIT_TIME:
    # Poll for messages indicating data flow or errors
    msg = test_bus.poll(Gst.MessageType.ANY, 100 * Gst.MSECOND)
    if msg:
        if msg.type == Gst.MessageType.ERROR:
            err, debug_info = msg.parse_error()
            print(f"Test pipeline error: {err}, {debug_info}")
            break
        if msg.type == Gst.MessageType.EOS:
            print("Test pipeline reached end-of-stream.")
            break
        if msg.type == Gst.MessageType.ELEMENT:
            # Some element-specific messages can indicate data flow
            struct = msg.get_structure()
            if struct and struct.has_name("GstUDPSrcTimeout"):
                print("UDP source timeout message received.")
            print(f"Got element message: {struct.get_name() if struct else 'unknown'}")

    # Try to observe data flow in a very simple way using a probe
    fakesink = test_pipeline.get_by_name('testfakesink')
    sink_pad = fakesink.get_static_pad('sink')
    if sink_pad:
        # Check if data is flowing through the pad
        position = test_pipeline.query_position(Gst.Format.TIME)[1]
        if position > 0:
            packet_received = True
            print(f"Packet received! Position: {position/Gst.SECOND:.3f}s")
            break
    
    # Short delay before checking again
    time.sleep(0.1)

# Stop test pipeline
test_pipeline.set_state(Gst.State.NULL)

if not packet_received:
    print("\n==== ERROR: No UDP packets received! ====")
    print("Troubleshooting steps:")
    print("1. Check that your camera is sending H.264 over RTP to this machine's IP address on port 5000")
    print("2. Verify Docker network configuration: container should use host networking (--net=host)")
    print("   or have port 5000/udp properly mapped to the host (-p 5000:5000/udp)")
    print("3. Check for firewall rules blocking UDP on port 5000")
    print("4. Try running 'nc -ul 5000' in another terminal to see if UDP packets arrive")
    print("5. Verify the camera's streaming settings\n")
    exit(1)
else:
    print("UDP packet received successfully! Continuing with full pipeline...")

# 3. Build the full pipeline with proper caps, jitterbuffer, decodebin, and appsink
print("Building full video processing pipeline...")
pipeline = Gst.parse_launch(
    'udpsrc port=5000 caps="application/x-rtp,media=video,'
    'clock-rate=90000,encoding-name=H264,payload=96" ! '
    'rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
    'video/x-raw,format=BGR ! appsink name=sink '
    'emit-signals=false sync=false max-buffers=1 drop=true'
)

# Get pipeline bus for message handling
bus = pipeline.get_bus()

# 4. Start playback
print("Attempting to set full pipeline to PLAYING...")
ret = pipeline.set_state(Gst.State.PLAYING)
print(f"Full pipeline state change result: {ret}")

# Retrieve the appsink for frame pulling
sink = pipeline.get_by_name('sink')

writer = None
start_time = None

if HEADLESS:
    start_time = time.time()
    print(f"Attempting to record video for {DURATION} seconds...")

# Add timeout for first frame
first_frame_received = False
first_frame_timeout = time.time() + MAX_WAIT_TIME

while True:
    sample = sink.emit('try-pull-sample', 500 * Gst.MSECOND)  # 500ms timeout
    
    # Check for timeout on first frame
    if not first_frame_received:
        if time.time() > first_frame_timeout:
            print(f"ERROR: No video frames received after {MAX_WAIT_TIME} seconds")
            print("Pipeline appears to be running but no decoded video frames available")
            break
    
    if not sample:
        # Check for messages on the bus
        msg = bus.poll(Gst.MessageType.ERROR | Gst.MessageType.EOS, 10 * Gst.MSECOND)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug_info = msg.parse_error()
                print(f"GStreamer Error: {err}, {debug_info}")
            elif msg.type == Gst.MessageType.EOS:
                print("End-Of-Stream reached.")
            break # Exit loop on error or EOS

        # Check if pipeline is still playing, otherwise EOS or error
        state, _, _ = pipeline.get_state(Gst.CLOCK_TIME_NONE)
        if state != Gst.State.PLAYING:
            print(f"Pipeline state changed to {state}, stopping.")
            break
        
        # If no sample but still running, try again
        if HEADLESS and start_time is not None and time.time() - start_time >= DURATION:
            print("Recording duration reached, but pipeline might be stalled.")
            break
        continue

    # We got a sample!
    if not first_frame_received:
        first_frame_received = True
        print("SUCCESS: First video frame received! Continuing with recording/display.")
    
    buf = sample.get_buffer()
    caps = sample.get_caps()
    struct = caps.get_structure(0)
    width = struct.get_value('width')
    height = struct.get_value('height')

    # Parse framerate using regex to avoid GstFraction errors
    fps = DEFAULT_FPS # Default value
    try:
        caps_string = caps.to_string() # Get caps as string
        # Use regex to find framerate=num/den
        match = re.search(r'framerate=(\d+)/(\d+)', caps_string)
        if match:
            num = int(match.group(1))
            den = int(match.group(2))
            if den != 0:
                fps = num / den
        else:
            print(f"Warning: Framerate not found in caps string: {caps_string}, using default {DEFAULT_FPS} FPS.")
    except Exception as e:
        print(f"Warning: Error parsing framerate from caps: {e}, using default {DEFAULT_FPS} FPS.")

    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        buf.unmap(mapinfo) # Should be called even on failure if mapinfo is valid
        continue
    
    frame = np.frombuffer(mapinfo.data, np.uint8).reshape((height, width, 3))
    buf.unmap(mapinfo)

    if HEADLESS:
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from XVID to mp4v
            filename = 'output.mp4'  # Changed from .avi to .mp4
            writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"Error: Failed to open VideoWriter for {filename} with {fps} FPS and resolution {width}x{height}")
                HEADLESS = False # Stop trying to write
                continue # Skip this frame
            else:
                print(f"Recording started to {filename} at {fps:.2f} FPS ({width}x{height}).")

        if writer.isOpened():
            writer.write(frame)

        # Check duration after processing frame
        if start_time is not None and time.time() - start_time >= DURATION:
            print(f"Recording finished after {DURATION} seconds.")
            break
    else:
        cv2.imshow('Optroxa', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Stop pipeline
pipeline.set_state(Gst.State.NULL)

# Flush bus messages if any remain
while bus.have_pending():
    bus.pop()

if writer is not None:
    print("Releasing video writer...")
    writer.release()
    print("Video writer released.")


if not HEADLESS:
    cv2.destroyAllWindows()

print("Script finished.")