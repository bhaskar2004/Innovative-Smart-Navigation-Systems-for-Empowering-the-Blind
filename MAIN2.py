import requests  # For making HTTP requests to APIs
import time  # For adding delays or managing timing in your application
import logging  # For logging important events or errors
import math  # For mathematical functions, such as angle calculations
import cv2  # For computer vision tasks (obstacle detection)
import numpy as np  # For handling numerical data and arrays
import polyline  # For encoding/decoding geographic coordinates (for navigation)
import speech_recognition as sr  # For speech-to-text (recognizing voice commands)
import pyttsx3  # For text-to-speech (giving spoken feedback)
from geopy.distance import geodesic  # For calculating distances between geographical points

class ObstacleDetector:
    def __init__(self, weights_path, config_path, names_path):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            self.confidence_threshold = 0.5
            self.nms_threshold = 0.4

        except Exception as e:
            self.logger.error(f"Error initializing obstacle detector: {e}")
            raise

    def detect_obstacles(self, frame):
        try:
            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)

            layer_outputs = self.net.forward(self.output_layers)

            detected_obstacles = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        obstacle = {
                            'class': self.classes[class_id],
                            'confidence': float(confidence),
                            'bbox': [x, y, w, h]
                        }
                        detected_obstacles.append(obstacle)

            return detected_obstacles

        except Exception as e:
            self.logger.error(f"Error in obstacle detection: {e}")
            return []

    def draw_detections(self, frame, detections):
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = f"{detection['class']}: {detection['confidence']:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

class NavigationSystem:
    def __init__(self, api_key, yolo_weights, yolo_config, yolo_names):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.api_key = api_key
        self.speaker = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.speaker.setProperty('rate', 150)
        self.speaker.setProperty('volume', 0.8)

        try:
            self.obstacle_detector = ObstacleDetector(
                weights_path=yolo_weights,
                config_path=yolo_config,
                names_path=yolo_names
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize obstacle detector: {e}")
            self.speak("Could not initialize obstacle detection system.")

    def get_coordinates_from_name(self, location_name):
        try:
            url = 'https://graphhopper.com/api/1/geocode'
            params = {
                'q': location_name,
                'key': self.api_key
            }
            response = requests.get(url, params=params)
            data = response.json()

            if 'hits' in data and len(data['hits']) > 0:
                lat = data['hits'][0]['point']['lat']
                lon = data['hits'][0]['point']['lng']
                return [lat, lon]
            else:
                self.logger.error("Error fetching coordinates for the location.")
                return None
        except Exception as e:
            self.logger.error(f"Geocoding error: {e}")
            return None

    def get_route(self, start_coords, end_coords):
        try:
            url = 'https://graphhopper.com/api/1/route'
            params = {
                'point': [f"{start_coords[0]},{start_coords[1]}", f"{end_coords[0]},{end_coords[1]}"],
                'vehicle': 'foot',
                'key': self.api_key
            }
            response = requests.get(url, params=params)
            data = response.json()

            if 'paths' in data:
                encoded_polyline = data['paths'][0]['points']
                decoded_points = polyline.decode(encoded_polyline)
                return decoded_points
            else:
                self.logger.error("Error fetching route from GraphHopper.")
                return None
        except Exception as e:
            self.logger.error(f"Route fetching error: {e}")
            return None

    def get_current_location(self):
        return [13.3957, 77.7270]

    def speak(self, message):
        try:
            print(f"Speaking: {message}")
            self.speaker.say(message)
            self.speaker.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech error: {e}")

    def listen_for_input(self):
        try:
            with sr.Microphone() as source:
                self.speak("Listening for your destination.")
                print("Listening...")

                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    return text

                except sr.UnknownValueError:
                    self.speak("Sorry, I could not understand that. Please try again.")
                    return None

                except sr.RequestError:
                    self.speak("Sorry, there was an error with the speech recognition service.")
                    return None

        except Exception as e:
            self.logger.error(f"Error in voice input: {e}")
            self.speak("An error occurred while listening for input.")
            return None

    def detect_and_alert_obstacles(self):
        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                self.speak("Could not open camera. Please check your camera connection.")
                return

            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.speak("Could not capture image from camera.")
                return

            obstacles = self.obstacle_detector.detect_obstacles(frame)

            critical_obstacles = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus']
            obstacle_positions = []

            for obstacle in obstacles:
                if obstacle['class'] in critical_obstacles and obstacle['confidence'] > 0.6:
                    # Get the center of the obstacle's bounding box
                    x, y, w, h = obstacle['bbox']
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Classify the obstacle's relative position
                    if center_x < frame.shape[1] // 3:  # Left side
                        position = 'left'
                    elif center_x > 2 * frame.shape[1] // 3:  # Right side
                        position = 'right'
                    elif center_y < frame.shape[0] // 3:  # Front (near)
                        position = 'front'
                    else:
                        position = 'back'  # Behind

                    obstacle_positions.append((obstacle['class'], position))

            # Now give instructions based on the positions of the obstacles
            for obstacle_class, position in obstacle_positions:
                if position == 'left':
                    self.speak(f"Warning: {obstacle_class} detected on the left. Slightly move right to avoid collision.")
                elif position == 'right':
                    self.speak(f"Warning: {obstacle_class} detected on the right. Slightly move left to avoid collision.")
                elif position == 'front':
                    self.speak(f"Warning: {obstacle_class} detected ahead. Move back to avoid collision.")
                elif position == 'back':
                    self.speak(f"Warning: {obstacle_class} detected front of you. Move left or right.")
        except Exception as e:
            self.logger.error(f"Obstacle detection error: {e}")
            self.speak("Error in obstacle detection system.")

    def calculate_direction(self, current_point, next_point):
        lat1, lon1 = current_point
        lat2, lon2 = next_point

        # Calculate bearing between two coordinates
        diff_lon = lon2 - lon1
        x = math.cos(math.radians(lat2)) * math.sin(math.radians(diff_lon))
        y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
            math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(diff_lon))

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)

        compass_bearing = (initial_bearing + 360) % 360

        if 45 <= compass_bearing < 135:
            return "turn right"
        elif 135 <= compass_bearing < 225:
            return "turn around"
        elif 225 <= compass_bearing < 315:
            return "turn left"
        else:
            return "continue straight"

    def navigate(self, destination_name):
        try:
            self.speak(f"Navigating to {destination_name}")

            start_coords = self.get_current_location()

            end_coords = self.get_coordinates_from_name(destination_name)

            if not end_coords:
                self.speak("Could not find coordinates for the destination.")
                return

            route = self.get_route(start_coords, end_coords)

            if not route:
                self.speak("Could not calculate route.")
                return

            for i in range(len(route) - 1):
                current_point = route[i]
                next_point = route[i + 1]

                distance = geodesic(current_point, next_point).meters

                direction = self.calculate_direction(current_point, next_point)

                self.speak(f"Walk {distance:.1f} meters. {direction}")

                self.detect_and_alert_obstacles()

                time.sleep(5)

            self.speak("You have reached your destination!")

        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            self.speak("Navigation encountered an error.")

def main():
    API_KEY = '9a1cc28f-e0a8-4025-8585-07c54f1f7c5f'  # Replace with your GraphHopper API key
    YOLO_WEIGHTS = 'C:/Users/Lenovo/OneDrive/Desktop/blind_assist/yolov3.weights/yolov3.weights'
    YOLO_CONFIG = 'C:/Users/Lenovo/OneDrive/Desktop/blind_assist/yolov3.cfg'
    YOLO_NAMES = 'C:/Users/Lenovo/OneDrive/Desktop/blind_assist/coco.names'

    nav_system = NavigationSystem(
        api_key=API_KEY, 
        yolo_weights=YOLO_WEIGHTS,
        yolo_config=YOLO_CONFIG,
        yolo_names=YOLO_NAMES
    )

    try:
        nav_system.speak("Welcome to the Blind Assistance Navigation System")
        nav_system.speak("Please tell me your destination")

        destination = nav_system.listen_for_input()

        if destination:
            nav_system.navigate(destination)
        else:
            nav_system.speak("No destination heard. Please try again.")

    except Exception as e:
        print(f"Error in main navigation: {e}")
        nav_system.speak("An error occurred during navigation.")

if __name__ == '__main__':
    main()
