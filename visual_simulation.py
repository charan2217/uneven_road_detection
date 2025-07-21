import pygame
import sys
import numpy as np
import time
import random
import math
from sensor_acquisition import SensorAcquisition
from sensor_fusion import SensorFusion

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 30

# Night Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
NIGHT_SKY = (25, 25, 112)  # Midnight blue
ROAD_NIGHT = (40, 40, 40)  # Dark asphalt
HEADLIGHT_YELLOW = (255, 255, 200)
DASHBOARD_BLUE = (0, 50, 100)

class NightRoadHazardSimulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI-Powered Nighttime Road Hazard Detection System")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 48)
        
        # Car properties
        self.car_x = 100
        self.car_y = SCREEN_HEIGHT // 2
        self.car_speed = 3
        self.normal_speed = 3
        self.slow_speed = 1
        self.car_moving = True
        
        # Road properties
        self.road_y = SCREEN_HEIGHT // 2 - 50
        self.road_height = 100
        self.road_offset = 0
        
        # Hazards that move relative to car
        self.hazards = []
        self.generate_hazards()
        
        # Real AI components
        self.sensor_acq = SensorAcquisition(mode="mock")
        self.sensor_fusion = SensorFusion(normalization=True)
        self.model = None
        self.load_model()
        
        # Metrics
        self.hazard_probability = 0.0
        self.hazard_detected = False
        self.sensor_readings = {}
        self.alert_timer = 0
        self.frame_count = 0
        
        # Demo statistics
        self.total_detections = 0
        self.hazards_avoided = 0
        self.distance_traveled = 0
        
        # Night effects
        self.headlight_range = 200
        self.stars = self.generate_stars()
        
    def load_model(self):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model("road_hazard_lstm_model.h5")
            print("✅ AI Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load AI model: {e}")
            print("🔄 Using intelligent simulation mode")
            self.model = None
            
    def generate_stars(self):
        stars = []
        for _ in range(100):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT // 2)
            brightness = random.randint(100, 255)
            stars.append((x, y, brightness))
        return stars
        
    def generate_hazards(self):
        # Generate hazards at different distances ahead
        hazard_distances = [400, 600, 800, 1000, 1200]
        hazard_types = ['pothole', 'crack', 'debris', 'pothole', 'crack']
        
        for i, distance in enumerate(hazard_distances):
            self.hazards.append({
                'x': distance,
                'y': self.road_y + random.randint(20, 70),
                'type': hazard_types[i],
                'detected': False,
                'size': random.randint(15, 30)
            })
    
    def get_real_ai_prediction(self):
        # Get real sensor data and AI prediction
        try:
            raw_data = self.sensor_acq.acquire_all(n_samples=1)
            fused_data = self.sensor_fusion.fuse(raw_data)
            
            # Store readings for display
            self.sensor_readings = {
                'Ultrasonic': round(raw_data['Ultrasonic(cm)'].iloc[0], 1),
                'Infrared': raw_data['Infrared'].iloc[0],
                'LiDAR': round(raw_data['LiDAR(cm)'].iloc[0], 1),
                'Thermal': round(raw_data['Thermal(C)'].iloc[0], 1)
            }
            
            # Real AI prediction
            if self.model:
                X = fused_data.values
                X = np.expand_dims(X, axis=1)
                self.hazard_probability = float(self.model.predict(X, verbose=0)[0][0])
                self.hazard_detected = self.hazard_probability > 0.5
            else:
                # Fallback intelligent simulation
                self.simulate_ai_prediction()
                
        except Exception as e:
            print(f"Error in AI prediction: {e}")
            self.simulate_ai_prediction()
    
    def simulate_ai_prediction(self):
        # Enhanced simulation when model isn't available
        hazard_nearby = False
        closest_hazard_distance = float('inf')
        
        for hazard in self.hazards:
            distance = hazard['x'] - self.car_x
            if 0 < distance < self.headlight_range:
                hazard_nearby = True
                closest_hazard_distance = min(closest_hazard_distance, distance)
                if not hazard['detected']:
                    hazard['detected'] = True
                    self.total_detections += 1
        
        # Simulate sensor readings based on proximity
        if hazard_nearby:
            proximity_factor = max(0, 1 - (closest_hazard_distance / self.headlight_range))
            base_ultrasonic = 12 + proximity_factor * 20
            base_thermal = 30 + proximity_factor * 15
            infrared = 0 if proximity_factor > 0.5 else 1
            base_lidar = 20 - proximity_factor * 10
        else:
            base_ultrasonic = random.uniform(12, 16)
            base_thermal = random.uniform(28, 35)
            infrared = 1
            base_lidar = random.uniform(15, 25)
        
        self.sensor_readings = {
            'Ultrasonic': round(base_ultrasonic, 1),
            'Infrared': infrared,
            'LiDAR': round(base_lidar, 1),
            'Thermal': round(base_thermal, 1)
        }
        
        # Calculate hazard probability
        hazard_score = 0
        if base_ultrasonic > 18: hazard_score += 0.3
        if infrared == 0: hazard_score += 0.25
        if base_lidar < 20: hazard_score += 0.2
        if base_thermal > 38: hazard_score += 0.25
        
        self.hazard_probability = max(0, min(1, hazard_score + random.uniform(-0.1, 0.1)))
        self.hazard_detected = self.hazard_probability > 0.5
    
    def update_vehicle_movement(self):
        # Adjust speed based on hazard detection
        if self.hazard_detected:
            self.car_speed = self.slow_speed
            self.alert_timer = 90  # 3 seconds at 30 FPS
            self.hazards_avoided += 1
        else:
            self.car_speed = self.normal_speed
        
        # Move hazards relative to car (simulate car movement)
        for hazard in self.hazards:
            hazard['x'] -= self.car_speed
            
        # Reset hazards that have passed
        for hazard in self.hazards:
            if hazard['x'] < -50:
                hazard['x'] = SCREEN_WIDTH + random.randint(200, 400)
                hazard['detected'] = False
                hazard['type'] = random.choice(['pothole', 'crack', 'debris'])
        
        # Update distance traveled
        self.distance_traveled += self.car_speed * 0.1  # Convert to meters
        
        # Update road animation
        self.road_offset = (self.road_offset + self.car_speed) % 80
    
    def draw_night_sky(self):
        # Night sky gradient
        for y in range(0, SCREEN_HEIGHT // 2):
            color_intensity = int(25 + (y / (SCREEN_HEIGHT // 2)) * 30)
            color = (color_intensity, color_intensity, min(112, color_intensity + 50))
            pygame.draw.line(self.screen, color, (0, y), (SCREEN_WIDTH, y))
        
        # Stars
        for star in self.stars:
            x, y, brightness = star
            # Twinkling effect
            twinkle = int(brightness + 50 * math.sin(self.frame_count * 0.1 + x * 0.01))
            twinkle = max(100, min(255, twinkle))
            star_color = (twinkle, twinkle, twinkle)
            pygame.draw.circle(self.screen, star_color, (x, y), 1)
    
    def draw_road(self):
        # Road surface
        pygame.draw.rect(self.screen, ROAD_NIGHT, 
                        (0, self.road_y, SCREEN_WIDTH, self.road_height))
        
        # Road edges
        pygame.draw.line(self.screen, WHITE, (0, self.road_y), (SCREEN_WIDTH, self.road_y), 3)
        pygame.draw.line(self.screen, WHITE, (0, self.road_y + self.road_height), 
                        (SCREEN_WIDTH, self.road_y + self.road_height), 3)
        
        # Animated center line
        for x in range(-self.road_offset, SCREEN_WIDTH, 80):
            pygame.draw.rect(self.screen, YELLOW, (x, self.road_y + 45, 40, 10))
    
    def draw_car_with_headlights(self):
        # Car body
        car_rect = pygame.Rect(self.car_x, self.car_y, 80, 40)
        color = RED if self.hazard_detected else BLUE
        pygame.draw.rect(self.screen, color, car_rect, border_radius=5)
        
        # Car details
        pygame.draw.rect(self.screen, (50, 50, 100), (self.car_x + 10, self.car_y + 5, 60, 15))  # Windshield
        
        # Headlights (bright circles)
        pygame.draw.circle(self.screen, HEADLIGHT_YELLOW, (self.car_x + 75, self.car_y + 10), 8)
        pygame.draw.circle(self.screen, HEADLIGHT_YELLOW, (self.car_x + 75, self.car_y + 30), 8)
        
        # Headlight beams
        beam_points = [
            (self.car_x + 80, self.car_y + 5),
            (self.car_x + self.headlight_range, self.car_y - 20),
            (self.car_x + self.headlight_range, self.car_y + 60),
            (self.car_x + 80, self.car_y + 35)
        ]
        
        # Create surface for headlight beam with transparency
        beam_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(beam_surface, (255, 255, 200, 30), beam_points)
        self.screen.blit(beam_surface, (0, 0))
        
        # Car wheels
        pygame.draw.circle(self.screen, BLACK, (self.car_x + 15, self.car_y + 45), 10)
        pygame.draw.circle(self.screen, BLACK, (self.car_x + 65, self.car_y + 45), 10)
        pygame.draw.circle(self.screen, GRAY, (self.car_x + 15, self.car_y + 45), 6)
        pygame.draw.circle(self.screen, GRAY, (self.car_x + 65, self.car_y + 45), 6)
        
        # Sensor visualization
        center_x, center_y = self.car_x + 40, self.car_y + 20
        if self.hazard_detected:
            for radius in [60, 80, 100]:
                pygame.draw.circle(self.screen, (255, 0, 0, 100), (center_x, center_y), radius, 2)
        else:
            for radius in [50, 70]:
                pygame.draw.circle(self.screen, (0, 255, 0, 50), (center_x, center_y), radius, 1)
    
    def draw_hazards(self):
        for hazard in self.hazards:
            x, y = hazard['x'], hazard['y']
            size = hazard['size']
            
            # Only draw hazards within headlight range or if detected
            distance_from_car = abs(x - self.car_x)
            in_headlight = distance_from_car < self.headlight_range
            
            if in_headlight or hazard['detected']:
                if hazard['type'] == 'pothole':
                    color = ORANGE if hazard['detected'] else BLACK
                    pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
                    pygame.draw.circle(self.screen, BLACK, (int(x), int(y)), size - 5)
                elif hazard['type'] == 'crack':
                    color = ORANGE if hazard['detected'] else (100, 100, 100)
                    pygame.draw.line(self.screen, color, (x, y - 20), (x + 40, y + 20), 8)
                elif hazard['type'] == 'debris':
                    color = ORANGE if hazard['detected'] else (139, 69, 19)
                    pygame.draw.rect(self.screen, color, (x, y - 10, size, size))
    
    def draw_night_dashboard(self):
        # Dashboard background with night theme
        dashboard_rect = pygame.Rect(SCREEN_WIDTH - 400, 10, 390, 380)
        pygame.draw.rect(self.screen, DASHBOARD_BLUE, dashboard_rect, border_radius=15)
        pygame.draw.rect(self.screen, WHITE, dashboard_rect, 3, border_radius=15)
        
        # Title
        title = self.title_font.render("🌙 Night AI Detection", True, WHITE)
        self.screen.blit(title, (SCREEN_WIDTH - 390, 25))
        
        # Hazard probability with glowing effect
        prob_text = f"Hazard Probability: {self.hazard_probability:.2f}"
        prob_color = RED if self.hazard_detected else GREEN
        prob_surface = self.font.render(prob_text, True, prob_color)
        self.screen.blit(prob_surface, (SCREEN_WIDTH - 390, 75))
        
        # Glowing progress bar
        bar_rect = pygame.Rect(SCREEN_WIDTH - 390, 100, 320, 25)
        pygame.draw.rect(self.screen, BLACK, bar_rect, border_radius=5)
        fill_width = int(320 * self.hazard_probability)
        if fill_width > 0:
            fill_rect = pygame.Rect(SCREEN_WIDTH - 390, 100, fill_width, 25)
            pygame.draw.rect(self.screen, prob_color, fill_rect, border_radius=5)
        pygame.draw.rect(self.screen, WHITE, bar_rect, 2, border_radius=5)
        
        # Status with night icons
        status = "🚨 HAZARD DETECTED!" if self.hazard_detected else "✅ Road Clear"
        status_color = RED if self.hazard_detected else GREEN
        status_surface = self.big_font.render(status, True, status_color)
        self.screen.blit(status_surface, (SCREEN_WIDTH - 390, 135))
        
        # Speed and distance
        speed_text = f"🚗 Speed: {self.car_speed} m/s"
        speed_surface = self.font.render(speed_text, True, WHITE)
        self.screen.blit(speed_surface, (SCREEN_WIDTH - 390, 175))
        
        distance_text = f"📏 Distance: {self.distance_traveled:.1f}m"
        distance_surface = self.font.render(distance_text, True, WHITE)
        self.screen.blit(distance_surface, (SCREEN_WIDTH - 390, 200))
        
        # Sensor readings with night theme
        y_offset = 230
        sensor_icons = {'Ultrasonic': '📡', 'Infrared': '🔴', 'LiDAR': '📊', 'Thermal': '🌡️'}
        for sensor, value in self.sensor_readings.items():
            icon = sensor_icons.get(sensor, '📊')
            if sensor == 'Infrared':
                text = f"{icon} {sensor}: {'Detected' if value else 'Clear'}"
            else:
                text = f"{icon} {sensor}: {value}"
            sensor_surface = self.font.render(text, True, WHITE)
            self.screen.blit(sensor_surface, (SCREEN_WIDTH - 390, y_offset))
            y_offset += 25
        
        # Statistics
        stats_y = 340
        stats_text = f"📈 Detected: {self.total_detections} | Avoided: {self.hazards_avoided}"
        stats_surface = self.font.render(stats_text, True, WHITE)
        self.screen.blit(stats_surface, (SCREEN_WIDTH - 390, stats_y))
    
    def draw_night_alert(self):
        if self.alert_timer > 0:
            # Pulsing red alert with night effect
            pulse = int(100 + 155 * abs(math.sin(self.frame_count * 0.2)))
            if self.alert_timer % 30 < 15:  # Flash effect
                alert_rect = pygame.Rect(50, 50, 400, 140)
                pygame.draw.rect(self.screen, (pulse, 0, 0), alert_rect, border_radius=15)
                pygame.draw.rect(self.screen, WHITE, alert_rect, 4, border_radius=15)
                
                alert_text = self.title_font.render("⚠️ NIGHT HAZARD!", True, WHITE)
                self.screen.blit(alert_text, (70, 80))
                
                action_text = self.big_font.render("🚗 Reducing speed...", True, WHITE)
                self.screen.blit(action_text, (70, 125))
            
            self.alert_timer -= 1
    
    def draw_project_info(self):
        # Project info banner
        info_rect = pygame.Rect(50, SCREEN_HEIGHT - 120, SCREEN_WIDTH - 100, 100)
        pygame.draw.rect(self.screen, (0, 0, 50, 200), info_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, info_rect, 2, border_radius=10)
        
        title = self.big_font.render("🌙 AI-Powered Nighttime Road Hazard Detection", True, WHITE)
        self.screen.blit(title, (70, SCREEN_HEIGHT - 110))
        
        subtitle = self.font.render("Real-time sensor fusion • LSTM Neural Network • Autonomous vehicle safety", True, YELLOW)
        self.screen.blit(subtitle, (70, SCREEN_HEIGHT - 80))
        
        tech = self.font.render("Technologies: Ultrasonic • Infrared • LiDAR • Thermal • TensorFlow • Python", True, WHITE)
        self.screen.blit(tech, (70, SCREEN_HEIGHT - 55))
    
    def run(self):
        running = True
        
        print("🌙 Starting Nighttime AI Road Hazard Detection Demo...")
        print("🚗 Vehicle will move and interact with hazards in real-time")
        print("🧠 Using real AI predictions from trained model")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Update AI prediction every 30 frames (1 second)
            if self.frame_count % 30 == 0:
                self.get_real_ai_prediction()
            
            # Update vehicle movement every frame
            self.update_vehicle_movement()
            
            # Draw everything
            self.draw_night_sky()
            self.draw_road()
            self.draw_hazards()
            self.draw_car_with_headlights()
            self.draw_night_dashboard()
            self.draw_night_alert()
            self.draw_project_info()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
            self.frame_count += 1
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    simulation = NightRoadHazardSimulation()
    simulation.run()
