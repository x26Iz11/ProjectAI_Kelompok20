import cv2
import threading
import tkinter as tk
from tkinter import simpledialog
import time
import os
from datetime import datetime
from flask import Flask, jsonify, request, make_response, send_from_directory
from functools import wraps


# Load pre-trained Haar Cascade face detection models for both frontal and side faces
face_cascade_front = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade_side = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")  # Correct path for side face detection

# Open the camera
camera = cv2.VideoCapture(0)

# Set a lower resolution for faster processing
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_lock = threading.Lock()
current_frame = None

# Lists for names and NIMs
names = []
nims = []

# Data structure for storing faces
face_database = {}

# Initialize attendance_data
attendance_data = []

# Initialize detected_names to track currently detected individuals
detected_names = set()

# Initialize next_id for unique IDs
next_id = 1

# Flask app setup
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure Unicode characters are handled correctly

# Timer variable for face detection
last_face_detected_time = time.time()
elapsed_time_without_face = 0

# Create directories for storing face and detected faces
if not os.path.exists("faces"):
    os.makedirs("faces")
if not os.path.exists("detected_faces"):
    os.makedirs("detected_faces")


def get_user_data():
    """Open input dialogs to collect names and NIMs."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    count = simpledialog.askinteger("Input", "Enter the number of people: ", minvalue=1)
    if count:
        for i in range(count):
            name = simpledialog.askstring("Input", f"Enter name for person {i + 1}:")
            nim = simpledialog.askstring("Input", f"Enter NIM for {name}:")
            if name and nim:
                names.append(name)
                nims.append(nim)
                capture_face_image(name, nim)  # Capture face for the user


def capture_face_image(name, nim):
    """Capture and save face images for a person."""
    global camera
    print(f"Please look at the camera, capturing face for {name}...")
    face_images = []
    for i in range(30):  # Capture 30 frames for training
        ret, frame = camera.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade_front.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_images.append(face_roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around face
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save captured face images to the database
    if len(face_images) > 0:
        face_database[name] = face_images
        print(f"Captured {len(face_images)} face images for {name}.")
    cv2.destroyAllWindows()


def face_detection(frame):
    """Detect faces in the frame using Haar Cascade (both frontal and side)."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect front faces
    faces_front = face_cascade_front.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    # Detect side faces
    faces_side = face_cascade_side.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    return faces_front, faces_side


def recognize_face(frame):
    """Recognize faces in the frame by comparing with the stored face database."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_front, faces_side = face_detection(frame)
    
    # Ensure faces_front and faces_side are lists (in case no faces are detected)
    faces_front = list(faces_front)
    faces_side = list(faces_side)

    all_faces = faces_front + faces_side  # Combine front and side faces

    recognized_person = None

    for (x, y, w, h) in all_faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        
        # Compare with the face database using template matching
        for name, stored_faces in face_database.items():
            for stored_face in stored_faces:
                try:
                    # Resize face_roi to match stored_face size
                    stored_face_resized = cv2.resize(stored_face, (face_roi.shape[1], face_roi.shape[0]))
                    similarity = cv2.matchTemplate(face_roi, stored_face_resized, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity)
                    if max_val > 0.7:  # If similarity is high enough
                        recognized_person = name
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        break
                except Exception as e:
                    print(f"Error comparing faces: {e}")
            if recognized_person:
                break

    return recognized_person


def save_attendance(name, nim, image_path):
    """Save detected names, NIM, and image to attendance_data."""
    global attendance_data, next_id
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract filename from image_path
    filename = os.path.basename(image_path)

    # Create a new attendance record with unique id
    new_record = {
        "id": next_id,  # Assign unique ID
        "name": name,
        "nim": nim,
        "time": now,
        "image": filename  # Save only the filename
    }

    # Append the new record to attendance_data
    attendance_data.append(new_record)
    print(f"Added to attendance_data: {new_record}")

    # Increment next_id for the next entry
    next_id += 1


def delete_face_data(name):
    """Delete face data for a given name, including images and from memory.""" 
    if name in face_database:
        del face_database[name]  # Remove from face database
        # Remove associated images
        for file in os.listdir("detected_faces"):
            if file.startswith(name + "_"):
                os.remove(os.path.join("detected_faces", file))
        print(f"Deleted face data for {name}.")
    else:
        print(f"No face data found for {name}.")


def drawer_box(frame):
    """Draw rectangles around detected faces and implement auto-focus behavior."""
    global last_face_detected_time, elapsed_time_without_face, attendance_data, detected_names

    # Detect faces
    faces_front, faces_side = face_detection(frame)
    
    # Ensure faces_front and faces_side are lists (in case no faces are detected)
    faces_front = list(faces_front)
    faces_side = list(faces_side)

    all_faces = faces_front + faces_side  # Combine front and side faces

    current_detected = set()

    if len(all_faces) > 0:
        last_face_detected_time = time.time()
        elapsed_time_without_face = 0  # Reset timer for face detection

        # Recognize faces
        for (x, y, w, h) in all_faces:
            recognized_person = recognize_face(frame)
            if recognized_person:
                current_detected.add(recognized_person)

        # Add new attendance entries only if name is not already detected
        for name in current_detected:
            if name not in detected_names:
                if name in names:
                    idx = names.index(name)
                    nim = nims[idx]
                    timestamp = int(time.time())
                    filename = f"{name}_{nim}_{timestamp}.jpg"
                    image_path = os.path.join("detected_faces", filename)
                    cv2.imwrite(image_path, frame)  # Save the frame with the detected face
                    save_attendance(name, nim, image_path)  # Save to attendance_data
                    detected_names.add(name)  # Mark as detected
    else:
        elapsed_time_without_face = time.time() - last_face_detected_time

    # Update detected_names: remove names that are no longer detected
    names_to_remove = detected_names - current_detected
    if names_to_remove:
        detected_names -= names_to_remove
        print(f"Names removed from detected_names: {names_to_remove}")

    # Draw rectangles around detected faces
    for (x, y, w, h) in all_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for face detection


def close_window():
    """Release the camera and close all OpenCV windows."""
    camera.release()
    cv2.destroyAllWindows()


def capture_frame():
    """Capture frames continuously and update the global frame."""
    global current_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        with frame_lock:
            current_frame = frame


def add_cors_headers(f):
    """Decorator to add CORS headers to responses."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        if isinstance(response, tuple):
            response, status = response
            resp = make_response(response, status)
        else:
            resp = make_response(response)
        resp.headers["Access-Control-Allow-Origin"] = "*"  # Allow all origins (change as needed)
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp
    return decorated_function


@app.route("/detected_faces/<path:filename>", methods=["GET"])
@add_cors_headers
def serve_image(filename):
    """Serve image files from detected_faces directory."""
    return send_from_directory('detected_faces', filename)


@app.route("/get_attendance", methods=["GET", "OPTIONS"])
@add_cors_headers
def get_attendance():
    if request.method == "OPTIONS":
        return '', 200
    print(f"Attendance data being sent: {attendance_data}")  # Logging for debugging
    return jsonify(attendance_data)


@app.route("/delete_attendance", methods=["POST", "OPTIONS"])
@add_cors_headers
def delete_attendance_route():
    if request.method == "OPTIONS":
        return '', 200
    data = request.get_json()
    id = data.get("id")
    try:
        id = int(id)  # Convert ID to integer
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid ID."}), 400
    global attendance_data
    print(f"Received delete request for ID: {id}")
    print(f"Current attendance_data IDs: {[item['id'] for item in attendance_data]}")
    # Find the entry with given id
    entry = next((item for item in attendance_data if item["id"] == id), None)
    if not entry:
        return jsonify({"status": "error", "message": "Data not found."}), 404
    # Remove the entry
    attendance_data = [item for item in attendance_data if item["id"] != id]
    print(f"Attendance data after deletion: {attendance_data}")
    # Delete the image file associated with the entry
    image_path = os.path.join("detected_faces", entry["image"])
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Deleted image file: {image_path}")
        except Exception as e:
            print(f"Failed to delete image file: {image_path}. Reason: {e}")
    return jsonify({"status": "success", "message": "Attendance data deleted successfully."})


@app.route("/reset_attendance", methods=["POST", "OPTIONS"])
@add_cors_headers
def reset_attendance_route():
    if request.method == "OPTIONS":
        return '', 200
    global attendance_data, next_id
    attendance_data = []
    detected_names.clear()  # Also reset detected_names
    next_id = 1  # Reset next_id
    print("All attendance data has been reset.")  # Logging for debugging

    # Hapus semua gambar di direktori 'detected_faces'
    for filename in os.listdir("detected_faces"):
        file_path = os.path.join("detected_faces", filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    return jsonify({"status": "success", "message": "All attendance data reset successfully."})


def main():
    # Get user data (names and NIMs)
    get_user_data()

    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_frame, daemon=True)
    capture_thread.start()

    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(debug=True, use_reloader=False), daemon=True)
    flask_thread.start()

    # Create a named window for the video feed
    window_name = "Face Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
                drawer_box(frame)
                cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break

    close_window()


if __name__ == "__main__":
    main()
