import os
import sys
import logging
import math
import time
import timm
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import speech_recognition as sr
import pyttsx3
import requests
import sounddevice as sd
import soundfile as sf
import polyline
import geopy.distance

from ultralytics import YOLO

class ConfigManager:
    """Manages configuration settings for the navigation system."""
    @staticmethod
    def get_config():
        return {
            # Replace with your actual GraphHopper API key
            'graphhopper_api_key': '9a1cc28f-e0a8-4025-8585-07c54f1f7c5f',
            # Default YOLO model path
            'yolo_model_path': 'yolov8n.pt',
            # Logging configuration
            'log_level': 'INFO',
            # Debug mode
            'debug_mode': False
        }

# The rest of the code remains the same as in the original file
class LoggerFactory:
    """Creates configured loggers with specified logging levels."""
    @staticmethod
    def create_logger(name: str, level: str = 'INFO') -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

class AdvancedObstacleDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Advanced obstacle detection using YOLOv8 with additional features
        """
        self.logger = LoggerFactory.create_logger(__name__)
        
        try:
            self.yolo_model = YOLO(model_path)
            
            # Optional: Load depth estimation model
            self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.depth_model.eval()
            
        except Exception as e:
            self.logger.error(f"Error initializing obstacle detector: {e}")
            raise

    def detect_obstacles(self, frame: np.ndarray) -> List[Dict]:
        """
        Advanced obstacle detection with multiple features
        """
        try:
            # YOLO Detection
            results = self.yolo_model(frame)[0]
            
            # Depth estimation
            depth_frame = self._estimate_depth(frame)
            
            detected_obstacles = []
            for result in results.boxes:
                # Convert tensor to numpy
                box = result.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box[:4]
                
                # Extract obstacle details
                obstacle = {
                    'class': self.yolo_model.names[int(result.cls)],
                    'confidence': float(result.conf),
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    'distance': self._calculate_distance(depth_frame, x1, y1, x2, y2)
                }
                
                detected_obstacles.append(obstacle)
            
            return detected_obstacles
        
        except Exception as e:
            self.logger.error(f"Obstacle detection error: {e}")
            return []

    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS model"""
        try:
            input_batch = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            input_batch = input_batch.unsqueeze(0)
            
            with torch.no_grad():
                depth_prediction = self.depth_model(input_batch)
            
            depth_map = depth_prediction.squeeze().cpu().numpy()
            return depth_map
        except Exception as e:
            self.logger.error(f"Depth estimation error: {e}")
            return np.zeros_like(frame[:,:,0], dtype=float)

    def _calculate_distance(self, depth_map: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate estimated distance to an object using depth map"""
        try:
            # Extract depth values from bounding box region
            roi_depth = depth_map[int(y1):int(y2), int(x1):int(x2)]
            
            # Average depth (lower value = closer object)
            avg_depth = np.mean(roi_depth)
            
            # Convert to meters (rough estimation)
            return float(avg_depth * 10)  # Calibration factor
        except Exception as e:
            self.logger.error(f"Distance calculation error: {e}")
            return float('inf')

class MultisensoryInterface:
    def __init__(self):
        """
        Advanced sensory interface for navigation guidance
        """
        self.logger = LoggerFactory.create_logger(__name__)
        self.speech_recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
        # Configure speech settings
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.8)

    def speak(self, message: str):
        """Text-to-speech output with logging"""
        try:
            self.logger.info(f"Speaking: {message}")
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech error: {e}")

    def listen(self, timeout: int = 5) -> Optional[str]:
        """
        Advanced speech recognition with multiple fallback mechanisms
        """
        try:
            with sr.Microphone() as source:
                self.speak("Listening for your destination.")
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
                
                try:
                    audio = self.speech_recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)
                    text = self.speech_recognizer.recognize_google(audio)
                    self.logger.info(f"Recognized destination: {text}")
                    return text
                
                except sr.UnknownValueError:
                    self.speak("Could not understand. Please try again.")
                except sr.RequestError:
                    self.speak("Speech service is unavailable.")
                
                return None
        except Exception as e:
            self.logger.error(f"Listening error: {e}")
            return None

    def provide_haptic_feedback(self, intensity: float = 0.5):
        """Generate haptic feedback for navigation guidance"""
        try:
            duration = 0.5  # seconds
            frequency = 100 * intensity  # Hz
            samples = np.sin(2 * np.pi * frequency * np.linspace(0, duration, int(44100 * duration), False))
            
            sd.play(samples, 44100)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Haptic feedback error: {e}")

class NavigationSystem:
    def __init__(self, 
                 api_key: str, 
                 obstacle_detector: AdvancedObstacleDetector,
                 sensory_interface: MultisensoryInterface):
        """
        Advanced navigation system with obstacle avoidance and guidance
        """
        self.logger = LoggerFactory.create_logger(__name__)
        self.api_key = api_key
        self.obstacle_detector = obstacle_detector
        self.sensory = sensory_interface
        
        # Navigation parameters
        self.safety_buffer = 1.5  # meters
        self.route_history = []

    def get_coordinates(self, location_name: str) -> Optional[List[float]]:
        """Geocode location name to coordinates"""
        try:
            url = 'https://graphhopper.com/api/1/geocode'
            params = {
                'q': location_name,
                'key': self.api_key
            }
            response = requests.get(url, params=params)
            data = response.json()

            if 'hits' in data and data['hits']:
                location = data['hits'][0]['point']
                return [location['lat'], location['lng']]
            
            self.sensory.speak(f"Could not find coordinates for {location_name}")
            return None
        
        except Exception as e:
            self.logger.error(f"Geocoding error: {e}")
            self.sensory.speak("Error finding location coordinates")
            return None

    def get_route(self, start: List[float], end: List[float]) -> Optional[List[Tuple[float, float]]]:
        """Retrieve route between two points"""
        try:
            url = 'https://graphhopper.com/api/1/route'
            params = {
                'point': [f"{start[0]},{start[1]}", f"{end[0]},{end[1]}"],
                'vehicle': 'foot',
                'key': self.api_key
            }
            response = requests.get(url, params=params)
            data = response.json()

            if 'paths' in data:
                encoded_route = data['paths'][0]['points']
                route = polyline.decode(encoded_route)
                return route
            
            self.sensory.speak("Could not calculate route")
            return None
        
        except Exception as e:
            self.logger.error(f"Route calculation error: {e}")
            self.sensory.speak("Navigation route calculation failed")
            return None

    def navigate(self, destination: str):
        """
        Main navigation workflow with obstacle detection and guidance
        """
        try:
            # Get current location (simulated for this example)
            current_location = [12.9716, 77.5946]  # Bangalore coordinates
            
            # Get destination coordinates
            destination_coords = self.get_coordinates(destination)
            
            if not destination_coords:
                return
            
            # Get route
            route = self.get_route(current_location, destination_coords)
            
            if not route:
                return
            
            self.sensory.speak(f"Navigating to {destination}")
            
            # Navigate through route waypoints
            for i in range(len(route) - 1):
                current_point = route[i]
                next_point = route[i + 1]
                
                # Obstacle detection
                frame = self._capture_frame()
                obstacles = self.obstacle_detector.detect_obstacles(frame)
                
                # Assess and handle obstacles
                self._handle_obstacles(obstacles, current_point, next_point)
                
                # Calculate distance to next waypoint
                distance = geopy.distance.distance(current_point, next_point).meters
                
                # Provide navigation guidance
                self._provide_navigation_guidance(current_point, next_point, distance)
                
                time.sleep(2)  # Pause between waypoints
            
            self.sensory.speak("You have reached your destination!")
        
        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            self.sensory.speak("Navigation encountered an unexpected error")

    def _capture_frame(self) -> np.ndarray:
        """Capture video frame for obstacle detection"""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else np.zeros((480, 640, 3), dtype=np.uint8)

    def _handle_obstacles(self, 
                      obstacles: List[Dict], 
                      current_point: Tuple[float, float], 
                      next_point: Tuple[float, float]):
      """Handle detected obstacles with detailed guidance."""
    
      high_risk_obstacles = [
        obs for obs in obstacles 
        if obs['class'] in ['car', 'truck', 'bicycle'] and 
        obs['distance'] < self.safety_buffer
      ]
    
      if high_risk_obstacles:
          self.sensory.speak("Caution: Obstacles detected.")
        
          for obstacle in high_risk_obstacles:
              obstacle_class = obstacle['class']
              obstacle_distance = obstacle['distance']
              obstacle_position = obstacle['position']  # Expecting relative position (e.g., 'left', 'right', 'front')

              # Provide directional guidance
              if obstacle_position == 'left':
                direction = "Move slightly to the right to avoid the obstacle."
              elif obstacle_position == 'right':
                direction = "Move slightly to the left to avoid the obstacle."
              elif obstacle_position == 'front':
                direction = "Stop and wait or move cautiously."
              else:
                direction = "Exercise caution around the obstacle."
            
                # Alert the user about the obstacle
                self.sensory.speak(f"Warning: {obstacle_class} detected {obstacle_distance} meters ahead on your {obstacle_position}.")
                self.sensory.speak(direction)
                self.sensory.provide_haptic_feedback()


    def _provide_navigation_guidance(self, 
                                     current_point: Tuple[float, float], 
                                     next_point: Tuple[float, float], 
                                     distance: float):
        """Generate contextual navigation instructions"""
        # Calculate bearing
        bearing = self._calculate_bearing(current_point, next_point)
        
        instruction = f"Walk {distance:.1f} meters "
        
        if 0 <= bearing < 45:
            instruction += "heading north"
        elif 45 <= bearing < 135:
            instruction += "heading east"
        elif 135 <= bearing < 225:
            instruction += "heading south"
        else:
            instruction += "heading west"
        
        self.sensory.speak(instruction)

    def _calculate_bearing(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate bearing between two geographic points"""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dL = lon2 - lon1
        X = math.cos(lat2) * math.sin(dL)
        Y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dL)
        
        bearing = math.degrees(math.atan2(X, Y))
        return (bearing + 360) % 360

def main():
    # Load configuration
    config = ConfigManager.get_config()
    
    # Initialize system components
    try:
        obstacle_detector = AdvancedObstacleDetector(
            model_path=config['yolo_model_path']
        )
        sensory_interface = MultisensoryInterface()
        
        navigation_system = NavigationSystem(
            api_key=config['graphhopper_api_key'],
            obstacle_detector=obstacle_detector,
            sensory_interface=sensory_interface
        )
        
        # Welcome message
        sensory_interface.speak("Welcome to Advanced Blind Assistance Navigation")
        
        # Get destination
        destination = sensory_interface.listen()
        
        if destination:
            navigation_system.navigate(destination)
        else:
            sensory_interface.speak("No destination received. Exiting.")
    
    except Exception as e:
        logging.error(f"System initialization error: {e}")
        sensory_interface.speak("System initialization failed")


if __name__ == '__main__':
    main()