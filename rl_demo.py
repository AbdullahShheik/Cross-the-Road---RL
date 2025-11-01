"""
Demo script showing how to extend the crossroad environment for RL
This file demonstrates the basic structure for future RL integration
"""

import time
from intersection_env import CrossroadEnvironment

class RLCrossroadDemo:
    def __init__(self):
        """
        Initialize the RL-ready environment
        This will be extended with RL capabilities in the future
        """
        self.env = CrossroadEnvironment()
        self.pedestrian_position = [-2.5, 3.5, 0.6]  # Starting position
        self.target_position = [2.5, 3.5, 0.6]       # Target: other side of road
        
    def get_state(self):
        """
        Get the current state for RL agent
        Future implementation will include:
        - Pedestrian position
        - Car positions and velocities  
        - Traffic light states
        - Distance to target
        """
        state = {
            'pedestrian_pos': self.pedestrian_position,
            'target_pos': self.target_position,
            'traffic_lights': self.env.signal_states,
            'cars': []
        }
        
        # Get car information
        for car in self.env.cars:
            import pybullet as p
            pos, orn = p.getBasePositionAndOrientation(car['body'])
            state['cars'].append({
                'position': pos,
                'direction': car['direction'],
                'speed': car['speed']
            })
            
        return state
    
    def calculate_reward(self, action, state):
        """
        Calculate reward for RL agent
        Future implementation will include:
        - Positive reward for reaching target
        - Negative reward for collision
        - Small negative reward for time taken
        - Bonus for safe crossing
        """
        # Placeholder reward function
        reward = 0
        done = False
        
        # Check if reached target (simplified)
        distance_to_target = abs(state['pedestrian_pos'][0] - state['target_pos'][0])
        if distance_to_target < 0.5:
            reward = 100  # Reached target
            done = True
            
        return reward, done
    
    def step(self, action):
        """
        Take an action in the environment
        Actions: 0=stay, 1=move_forward, 2=move_back, 3=move_left, 4=move_right
        """
        # This is where the RL agent will control the pedestrian
        # For now, we just return the current state
        
        # Update environment
        self.env.update_traffic_lights()
        self.env.update_cars()
        
        # Get new state
        state = self.get_state()
        reward, done = self.calculate_reward(action, state)
        
        return state, reward, done
    
    def reset(self):
        """Reset environment for new episode"""
        self.pedestrian_position = [-2.5, 3.5, 0.6]
        return self.get_state()
    
    def run_demo(self):
        """Run a demonstration of the RL-ready environment"""
        print("🤖 RL Demo Mode - Future Integration Points:")
        print("1. State space: Pedestrian pos, car positions, traffic lights")
        print("2. Action space: Move forward/back/left/right or stay")
        print("3. Reward function: Safe crossing with time penalty")
        print("4. Episode termination: Reach target or collision")
        print("\n🚦 Current simulation shows the environment ready for RL...")
        
        try:
            step = 0
            while True:
                # Simulate RL step (without actual learning for now)
                state = self.get_state()
                
                # Print state information every few seconds
                if step % 180 == 0:
                    print(f"\nStep {step}:")
                    print(f"  Traffic Lights: {state['traffic_lights']}")
                    print(f"  Active Cars: {len(state['cars'])}")
                    print(f"  Pedestrian ready for RL at: {state['pedestrian_pos']}")
                
                # Update environment
                self.env.update_traffic_lights() 
                self.env.update_cars()
                
                # Step physics
                import pybullet as p
                p.stepSimulation()
                time.sleep(1/60)
                
                step += 1
                
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        finally:
            import pybullet as p
            p.disconnect()
            print("✅ Demo ended successfully")

if __name__ == "__main__":
    demo = RLCrossroadDemo()
    demo.run_demo()