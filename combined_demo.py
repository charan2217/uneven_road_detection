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
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
NIGHT_SKY = (25, 25, 112)
ROAD_NIGHT = (60, 60, 60)  # Lighter road color
HEADLIGHT_YELLOW = (255, 255, 200)
DASHBOARD_BLUE = (0, 50, 100)
NEURAL_BLUE = (0, 150, 255)
FUSION_GREEN = (0, 200, 100)

class ComprehensiveAIDemo:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI-Powered Road Hazard Detection - Complete System Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.med_font = pygame.font.Font(None, 22)
        self.big_font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 32)
        
        # Layout - Split screen
        self.sim_width = SCREEN_WIDTH // 2
        self.pipeline_width = SCREEN_WIDTH // 2
        
        # Car properties
        self.car_x = 100
        self.car_y = SCREEN_HEIGHT // 2
        self.car_speed = 3
        self.normal_speed = 3
        self.slow_speed = 1
        
        # Road properties
        self.road_y = SCREEN_HEIGHT // 2 - 50
        self.road_height = 100
        self.road_offset = 0
        
        # Hazards
        self.hazards = []
        self.generate_hazards()
        
        # AI components
        self.sensor_acq = SensorAcquisition(mode="mock")
        self.sensor_fusion = SensorFusion(normalization=True)
        self.model = None
        self.load_model()
        
        # Metrics and data
        self.hazard_probability = 0.0
        self.hazard_detected = False
        self.sensor_readings = {}
        self.fused_data = None
        self.raw_sensor_data = None
        self.alert_timer = 0
        self.frame_count = 0
        
        # Pipeline visualization data
        self.sensor_history = []
        self.prediction_history = []
        self.pipeline_step = 0  # 0: sensors, 1: fusion, 2: AI, 3: control
        self.step_timer = 0
        
        # Demo statistics
        self.total_detections = 0
        self.hazards_avoided = 0
        self.distance_traveled = 0
        
        # Visual effects
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
        for _ in range(50):
            x = random.randint(0, self.sim_width)
            y = random.randint(0, SCREEN_HEIGHT // 2)
            brightness = random.randint(100, 255)
            stars.append((x, y, brightness))
        return stars
        
    def generate_hazards(self):
        hazard_distances = [300, 500, 700, 900, 1100]
        hazard_types = ['pothole', 'crack', 'debris', 'pothole', 'crack']
        
        for i, distance in enumerate(hazard_distances):
            self.hazards.append({
                'x': distance,
                'y': self.road_y + random.randint(20, 70),
                'type': hazard_types[i],
                'detected': False,
                'size': random.randint(15, 30)
            })
    
    def get_ai_prediction_with_pipeline(self):
        """Get AI prediction and store pipeline data for visualization"""
        try:
            # Step 1: Raw sensor data
            self.raw_sensor_data = self.sensor_acq.acquire_all(n_samples=1)
            self.sensor_readings = {
                'Ultrasonic': round(self.raw_sensor_data['Ultrasonic(cm)'].iloc[0], 1),
                'Infrared': self.raw_sensor_data['Infrared'].iloc[0],
                'LiDAR': round(self.raw_sensor_data['LiDAR(cm)'].iloc[0], 1),
                'Thermal': round(self.raw_sensor_data['Thermal(C)'].iloc[0], 1)
            }
            
            # Step 2: Sensor fusion
            self.fused_data = self.sensor_fusion.fuse(self.raw_sensor_data)
            
            # Step 3: AI prediction
            if self.model:
                X = self.fused_data.values
                X = np.expand_dims(X, axis=1)
                self.hazard_probability = float(self.model.predict(X, verbose=0)[0][0])
                self.hazard_detected = self.hazard_probability > 0.5
            else:
                self.simulate_ai_prediction()
            
            # Store history for visualization
            self.sensor_history.append(list(self.sensor_readings.values()))
            self.prediction_history.append(self.hazard_probability)
            
            # Keep only last 20 readings
            if len(self.sensor_history) > 20:
                self.sensor_history.pop(0)
                self.prediction_history.pop(0)
                
        except Exception as e:
            print(f"Error in AI prediction: {e}")
            self.simulate_ai_prediction()
    
    def simulate_ai_prediction(self):
        """Fallback when model isn't available"""
        hazard_nearby = False
        closest_distance = float('inf')
        
        for hazard in self.hazards:
            distance = hazard['x'] - self.car_x
            if 0 < distance < self.headlight_range:
                hazard_nearby = True
                closest_distance = min(closest_distance, distance)
                if not hazard['detected']:
                    hazard['detected'] = True
                    self.total_detections += 1
        
        if hazard_nearby:
            proximity = max(0, 1 - (closest_distance / self.headlight_range))
            base_ultrasonic = 12 + proximity * 20
            base_thermal = 30 + proximity * 15
            infrared = 0 if proximity > 0.5 else 1
            base_lidar = 20 - proximity * 10
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
        """Update vehicle movement and hazard positions"""
        if self.hazard_detected:
            self.car_speed = self.slow_speed
            self.alert_timer = 90
            self.hazards_avoided += 1
        else:
            self.car_speed = self.normal_speed
        
        # Move hazards
        for hazard in self.hazards:
            hazard['x'] -= self.car_speed
            
        # Reset passed hazards
        for hazard in self.hazards:
            if hazard['x'] < -50:
                hazard['x'] = self.sim_width + random.randint(200, 400)
                hazard['detected'] = False
                hazard['type'] = random.choice(['pothole', 'crack', 'debris'])
        
        self.distance_traveled += self.car_speed * 0.1
        self.road_offset = (self.road_offset + self.car_speed) % 80
        
        # Update pipeline step animation
        self.step_timer += 1
        if self.step_timer >= 30:  # Change step every second
            self.pipeline_step = (self.pipeline_step + 1) % 4
            self.step_timer = 0
    
    def draw_simulation_side(self):
        """Draw the left side - road simulation"""
        # Clip to left half
        clip_rect = pygame.Rect(0, 0, self.sim_width, SCREEN_HEIGHT)
        self.screen.set_clip(clip_rect)
        
        # Night sky gradient
        for y in range(0, SCREEN_HEIGHT):
            if y < SCREEN_HEIGHT // 2:
                intensity = int(25 + (y / (SCREEN_HEIGHT // 2)) * 30)
                color = (intensity, intensity, min(112, intensity + 50))
            else:
                # Ground area - darker
                intensity = int(15 + ((y - SCREEN_HEIGHT // 2) / (SCREEN_HEIGHT // 2)) * 20)
                color = (intensity, intensity + 5, intensity + 10)
            pygame.draw.line(self.screen, color, (0, y), (self.sim_width, y))
        
        # Stars
        for star in self.stars:
            x, y, brightness = star
            twinkle = int(brightness + 50 * math.sin(self.frame_count * 0.1 + x * 0.01))
            twinkle = max(100, min(255, twinkle))
            pygame.draw.circle(self.screen, (twinkle, twinkle, twinkle), (x, y), 2)
        
        # Road with better visibility
        pygame.draw.rect(self.screen, ROAD_NIGHT, (0, self.road_y, self.sim_width, self.road_height))
        
        # Road edges - brighter
        pygame.draw.line(self.screen, WHITE, (0, self.road_y), (self.sim_width, self.road_y), 4)
        pygame.draw.line(self.screen, WHITE, (0, self.road_y + self.road_height), 
                        (self.sim_width, self.road_y + self.road_height), 4)
        
        # Animated road lines - brighter and larger
        for x in range(-self.road_offset, self.sim_width, 80):
            pygame.draw.rect(self.screen, YELLOW, (x, self.road_y + 45, 50, 12))
        
        # Car with better visibility
        car_rect = pygame.Rect(self.car_x, self.car_y, 80, 40)
        car_color = RED if self.hazard_detected else (0, 150, 255)  # Brighter blue
        pygame.draw.rect(self.screen, car_color, car_rect, border_radius=5)
        
        # Car outline for better visibility
        pygame.draw.rect(self.screen, WHITE, car_rect, 3, border_radius=5)
        
        # Car windshield
        pygame.draw.rect(self.screen, (100, 150, 200), (self.car_x + 10, self.car_y + 5, 60, 15))
        
        # Headlights - brighter and larger
        pygame.draw.circle(self.screen, HEADLIGHT_YELLOW, (self.car_x + 75, self.car_y + 10), 10)
        pygame.draw.circle(self.screen, HEADLIGHT_YELLOW, (self.car_x + 75, self.car_y + 30), 10)
        pygame.draw.circle(self.screen, WHITE, (self.car_x + 75, self.car_y + 10), 6)
        pygame.draw.circle(self.screen, WHITE, (self.car_x + 75, self.car_y + 30), 6)
        
        # Headlight beam - more visible
        beam_points = [
            (self.car_x + 80, self.car_y + 5),
            (min(self.car_x + self.headlight_range, self.sim_width), self.car_y - 30),
            (min(self.car_x + self.headlight_range, self.sim_width), self.car_y + 70),
            (self.car_x + 80, self.car_y + 35)
        ]
        beam_surface = pygame.Surface((self.sim_width, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(beam_surface, (255, 255, 200, 60), beam_points)
        self.screen.blit(beam_surface, (0, 0))
        
        # Car wheels - more visible
        pygame.draw.circle(self.screen, BLACK, (self.car_x + 15, self.car_y + 45), 12)
        pygame.draw.circle(self.screen, BLACK, (self.car_x + 65, self.car_y + 45), 12)
        pygame.draw.circle(self.screen, GRAY, (self.car_x + 15, self.car_y + 45), 8)
        pygame.draw.circle(self.screen, GRAY, (self.car_x + 65, self.car_y + 45), 8)
        
        # Hazards with better visibility
        for hazard in self.hazards:
            x, y = hazard['x'], hazard['y']
            if x < self.sim_width:  # Only draw if in simulation area
                size = hazard['size']
                distance_from_car = abs(x - self.car_x)
                in_headlight = distance_from_car < self.headlight_range
                
                if in_headlight or hazard['detected']:
                    if hazard['type'] == 'pothole':
                        color = ORANGE if hazard['detected'] else (50, 50, 50)
                        pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
                        pygame.draw.circle(self.screen, BLACK, (int(x), int(y)), size - 5)
                    elif hazard['type'] == 'crack':
                        color = ORANGE if hazard['detected'] else (120, 120, 120)
                        pygame.draw.line(self.screen, color, (x, y - 20), (x + 40, y + 20), 10)
                    elif hazard['type'] == 'debris':
                        color = ORANGE if hazard['detected'] else (160, 100, 50)
                        pygame.draw.rect(self.screen, color, (x, y - 10, size, size))
        
        # Alert overlay
        if self.alert_timer > 0:
            pulse = int(100 + 155 * abs(math.sin(self.frame_count * 0.2)))
            if self.alert_timer % 30 < 15:
                alert_rect = pygame.Rect(20, 50, 350, 120)
                pygame.draw.rect(self.screen, (pulse, 0, 0), alert_rect, border_radius=10)
                pygame.draw.rect(self.screen, WHITE, alert_rect, 4, border_radius=10)
                
                alert_text = self.big_font.render("⚠️ HAZARD DETECTED!", True, WHITE)
                self.screen.blit(alert_text, (40, 75))
                
                action_text = self.med_font.render("🚗 Reducing speed...", True, WHITE)
                self.screen.blit(action_text, (40, 110))
            
            self.alert_timer -= 1
        
        # Simulation title
        title_rect = pygame.Rect(10, 10, self.sim_width - 20, 40)
        pygame.draw.rect(self.screen, (0, 0, 50, 220), title_rect, border_radius=5)
        pygame.draw.rect(self.screen, WHITE, title_rect, 2, border_radius=5)
        title = self.big_font.render("🌙 Nighttime Road Simulation", True, WHITE)
        self.screen.blit(title, (20, 20))
        
        # Remove clipping
        self.screen.set_clip(None)
    
    def draw_pipeline_side(self):
        """Draw the right side - AI pipeline visualization"""
        # Background
        pipeline_rect = pygame.Rect(self.sim_width, 0, self.pipeline_width, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, (30, 30, 50), pipeline_rect)
        
        # Vertical divider
        pygame.draw.line(self.screen, WHITE, (self.sim_width, 0), (self.sim_width, SCREEN_HEIGHT), 4)
        
        # Pipeline title
        title_rect = pygame.Rect(self.sim_width + 10, 10, self.pipeline_width - 20, 40)
        pygame.draw.rect(self.screen, DASHBOARD_BLUE, title_rect, border_radius=5)
        pygame.draw.rect(self.screen, WHITE, title_rect, 2, border_radius=5)
        title = self.big_font.render("🧠 AI Pipeline Process", True, WHITE)
        self.screen.blit(title, (self.sim_width + 20, 20))
        
        # Step indicators with better spacing
        step_y = 70
        step_height = 180
        steps = [
            ("1. Sensor Data", "📡", BLUE),
            ("2. Data Fusion", "🔄", FUSION_GREEN),
            ("3. AI Prediction", "🧠", NEURAL_BLUE),
            ("4. Vehicle Control", "🚗", ORANGE)
        ]
        
        for i, (step_name, icon, color) in enumerate(steps):
            y_pos = step_y + i * step_height
            
            # Highlight current step
            if i == self.pipeline_step:
                step_rect = pygame.Rect(self.sim_width + 10, y_pos - 10, self.pipeline_width - 20, step_height - 20)
                pygame.draw.rect(self.screen, (*color, 80), step_rect, border_radius=10)
                pygame.draw.rect(self.screen, color, step_rect, 3, border_radius=10)
            
            # Step title
            step_title = self.big_font.render(f"{icon} {step_name}", True, WHITE)
            self.screen.blit(step_title, (self.sim_width + 30, y_pos))
            
            # Step content
            if i == 0:  # Sensor Data
                self.draw_sensor_data(y_pos + 35)
            elif i == 1:  # Data Fusion
                self.draw_fusion_process(y_pos + 35)
            elif i == 2:  # AI Prediction
                self.draw_ai_prediction(y_pos + 35)
            elif i == 3:  # Vehicle Control
                self.draw_vehicle_control(y_pos + 35)
    
    def draw_sensor_data(self, y_pos):
        """Draw sensor data visualization"""
        x_start = self.sim_width + 40
        
        for i, (sensor, value) in enumerate(self.sensor_readings.items()):
            y = y_pos + i * 28
            
            # Sensor name
            sensor_text = self.med_font.render(f"{sensor}:", True, WHITE)
            self.screen.blit(sensor_text, (x_start, y))
            
            # Value with color coding
            if sensor == 'Infrared':
                value_text = "Detected" if value else "Clear"
                color = RED if value == 0 else GREEN
            else:
                value_text = str(value)
                # Color code based on danger level
                if sensor == 'Ultrasonic' and value > 18:
                    color = RED
                elif sensor == 'Thermal' and value > 38:
                    color = RED
                elif sensor == 'LiDAR' and value < 20:
                    color = RED
                else:
                    color = GREEN
            
            value_surface = self.med_font.render(value_text, True, color)
            self.screen.blit(value_surface, (x_start + 120, y))
    
    def draw_fusion_process(self, y_pos):
        """Draw sensor fusion visualization"""
        x_start = self.sim_width + 40
        
        # Fusion algorithm info
        fusion_text = self.med_font.render("Normalizing sensor data...", True, WHITE)
        self.screen.blit(fusion_text, (x_start, y_pos))
        
        # Show fused values if available
        if self.fused_data is not None and len(self.fused_data.columns) > 0:
            fused_values = self.fused_data.iloc[0].values
            for i, value in enumerate(fused_values[:4]):  # Show first 4 features
                y = y_pos + 25 + i * 25
                text = f"Feature {i+1}: {value:.3f}"
                value_surface = self.med_font.render(text, True, WHITE)
                self.screen.blit(value_surface, (x_start, y))
        else:
            # Fallback display
            for i in range(4):
                y = y_pos + 25 + i * 25
                value = random.uniform(0.1, 0.9)
                text = f"Feature {i+1}: {value:.3f}"
                value_surface = self.med_font.render(text, True, WHITE)
                self.screen.blit(value_surface, (x_start, y))
    
    def draw_ai_prediction(self, y_pos):
        """Draw AI prediction visualization"""
        x_start = self.sim_width + 40
        
        # Model info
        model_text = self.med_font.render("LSTM Neural Network", True, WHITE)
        self.screen.blit(model_text, (x_start, y_pos))
        
        # Prediction probability
        prob_text = f"Hazard Probability: {self.hazard_probability:.3f}"
        prob_color = RED if self.hazard_probability > 0.5 else GREEN
        prob_surface = self.med_font.render(prob_text, True, prob_color)
        self.screen.blit(prob_surface, (x_start, y_pos + 30))
        
        # Probability bar
        bar_rect = pygame.Rect(x_start, y_pos + 60, 250, 25)
        pygame.draw.rect(self.screen, GRAY, bar_rect)
        fill_width = int(250 * self.hazard_probability)
        if fill_width > 0:
            fill_rect = pygame.Rect(x_start, y_pos + 60, fill_width, 25)
            pygame.draw.rect(self.screen, prob_color, fill_rect)
        pygame.draw.rect(self.screen, WHITE, bar_rect, 2)
        
        # Decision
        decision = "HAZARD DETECTED!" if self.hazard_detected else "Road Clear"
        decision_surface = self.med_font.render(decision, True, prob_color)
        self.screen.blit(decision_surface, (x_start, y_pos + 95))
    
    def draw_vehicle_control(self, y_pos):
        """Draw vehicle control actions"""
        x_start = self.sim_width + 40
        
        # Current action
        if self.hazard_detected:
            action = "🚨 EMERGENCY BRAKING"
            action_color = RED
            speed_action = f"Speed: {self.car_speed} m/s (REDUCED)"
        else:
            action = "✅ NORMAL DRIVING"
            action_color = GREEN
            speed_action = f"Speed: {self.car_speed} m/s (NORMAL)"
        
        action_surface = self.med_font.render(action, True, action_color)
        self.screen.blit(action_surface, (x_start, y_pos))
        
        speed_surface = self.med_font.render(speed_action, True, WHITE)
        self.screen.blit(speed_surface, (x_start, y_pos + 30))
        
        # Statistics
        stats_text = f"Distance: {self.distance_traveled:.1f}m"
        stats_surface = self.med_font.render(stats_text, True, WHITE)
        self.screen.blit(stats_surface, (x_start, y_pos + 60))
        
        detection_text = f"Hazards Detected: {self.total_detections}"
        detection_surface = self.med_font.render(detection_text, True, WHITE)
        self.screen.blit(detection_surface, (x_start, y_pos + 90))
    
    def draw_project_info(self):
        """Draw project information at bottom"""
        info_rect = pygame.Rect(10, SCREEN_HEIGHT - 80, SCREEN_WIDTH - 20, 70)
        pygame.draw.rect(self.screen, (0, 0, 50, 240), info_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, info_rect, 3, border_radius=10)
        
        title = self.title_font.render("🌙 AI-Powered Nighttime Road Hazard Detection System", True, WHITE)
        self.screen.blit(title, (20, SCREEN_HEIGHT - 70))
        
        subtitle = self.med_font.render("Real-time Multi-sensor Fusion • LSTM Neural Network • Autonomous Safety Control", True, YELLOW)
        self.screen.blit(subtitle, (20, SCREEN_HEIGHT - 45))
        
        tech = self.font.render("Technologies: Ultrasonic • Infrared • LiDAR • Thermal • TensorFlow • Python • Pygame", True, WHITE)
        self.screen.blit(tech, (20, SCREEN_HEIGHT - 20))
    
    def run(self):
        """Main demo loop"""
        running = True
        
        print("🌙 Starting Comprehensive AI Road Hazard Detection Demo...")
        print("📺 Left side: Real-time road simulation")
        print("🧠 Right side: AI pipeline process visualization")
        print("🎥 Perfect for screen recording and LinkedIn demos!")
        print("Press SPACEBAR to reset statistics")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Reset demo
                        self.distance_traveled = 0
                        self.total_detections = 0
                        self.hazards_avoided = 0
                        print("Demo statistics reset!")
            
            # Update AI prediction every 30 frames (1 second)
            if self.frame_count % 30 == 0:
                self.get_ai_prediction_with_pipeline()
            
            # Update vehicle movement every frame
            self.update_vehicle_movement()
            
            # Clear screen
            self.screen.fill(BLACK)
            
            # Draw both sides
            self.draw_simulation_side()
            self.draw_pipeline_side()
            self.draw_project_info()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
            self.frame_count += 1
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    demo = ComprehensiveAIDemo()
    demo.run()
