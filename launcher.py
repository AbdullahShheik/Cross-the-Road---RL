#!/usr/bin/env python3
"""
Crossroad RL Environment Launcher
Simple interface for training and testing the pedestrian crossing agent
"""

import sys
import os
import argparse


def test_simulation():
    """Test the crossroad simulation environment"""
    print("Starting Crossroad Simulation Test...")
    print("This will show the 3D environment with cars and traffic lights.")
    print("Press Ctrl+C to stop.")
    print("-" * 50)
    
    from intersection_env import CrossroadEnvironment
    
    try:
        env = CrossroadEnvironment()
        env.run_simulation()
    except KeyboardInterrupt:
        print("\nTest completed!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


def train_model():
    """Train the DQN agent"""
    print("Starting DQN Agent Training...")
    print("This will train the pedestrian to navigate the roundabout safely.")
    print("Training may take a while depending on your hardware.")
    print("-" * 50)
    
    try:
        from train_agent import train_agent
        
        # Ask user for training parameters
        try:
            episodes = int(input("Enter number of training episodes (default 500): ") or "500")
        except ValueError:
            episodes = 500
            
        use_gui = input("Show GUI during training? (y/N): ").lower().startswith('y')
        
        print(f"\nStarting training for {episodes} episodes...")
        print(f"GUI: {'Enabled' if use_gui else 'Disabled (faster)'}")
        
        train_agent(
            num_episodes=episodes,
            max_steps=2000,
            gui=use_gui,
            save_dir='models',
            save_frequency=50
        )
        
        print("Training completed! Model saved in ./models/")
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_trained_model():
    """Run the trained agent"""
    model_path = 'models/dqn_model_final.pth'
    
    if not os.path.exists(model_path):
        print("No trained model found!")
        print("Please train the model first using option 2.")
        print(f"Looking for: {model_path}")
        return 1
    
    print("Running Trained Agent...")
    print("Watch the pedestrian navigate the roundabout!")
    print("The agent will cross multiple roads and avoid vehicles.")
    print("-" * 50)
    
    try:
        from evaluate_agent import evaluate_agent
        
        episodes = int(input("Enter number of episodes to run (default 5): ") or "5")
        
        results = evaluate_agent(
            model_path=model_path,
            num_episodes=episodes,
            gui=True,
            max_steps=2000
        )
        
        print("\nEvaluation completed!")
        print(f"Success rate: {results['success_rate']*100:.1f}%")
        return 0
        
    except KeyboardInterrupt:
        print("\nStopped by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Crossroad RL Environment - Train and test pedestrian crossing agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py test      # Test the simulation environment
  python launcher.py train     # Train the agent
  python launcher.py run       # Run trained agent
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['test', 'train', 'run'],
        nargs='?',
        help='Action to perform'
    )
    
    if len(sys.argv) == 1:
        # Interactive mode
        # NOTE: Avoid Unicode emojis in the banner for better Windows console compatibility
        print("=" * 60)
        print("CROSSROAD REINFORCEMENT LEARNING PROJECT")
        print("=" * 60)
        print("Train an AI agent to safely cross a busy roundabout!")
        print()
        print("Choose an option:")
        print("1. Test Simulation - View the 3D environment")
        print("2. Train Model - Train the AI agent")
        print("3. Run Model - Watch trained agent in action")
        print("4. Exit")
        print()
        
        while True:
            try:
                choice = input("Select option (1-4): ").strip()
                
                if choice == '1':
                    return test_simulation()
                elif choice == '2':
                    return train_model()
                elif choice == '3':
                    return run_trained_model()
                elif choice == '4':
                    print("Goodbye!")
                    return 0
                else:
                    print("Invalid choice. Please select 1-4.")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return 0
    else:
        # Command line mode
        args = parser.parse_args()
        
        if args.mode == 'test':
            return test_simulation()
        elif args.mode == 'train':
            return train_model()
        elif args.mode == 'run':
            return run_trained_model()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())