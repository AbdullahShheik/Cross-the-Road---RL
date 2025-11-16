#!/usr/bin/env python3
"""
Crossroad Environment Launcher
Provides easy access to different simulation modes
"""

import sys
import argparse

def run_basic_simulation():
    """Run the basic crossroad simulation"""
    print("🚦 Starting Basic Crossroad Simulation...")
    from intersection_env import CrossroadEnvironment
    
    env = CrossroadEnvironment()
    env.run_simulation()

def run_rl_demo():
    """Run the RL demonstration"""
    print("🤖 Starting RL Demo Mode...")
    from rl_demo import RLCrossroadDemo
    
    demo = RLCrossroadDemo()
    demo.run_demo()

def run_agent_demo():
    """Run agent control demonstration"""
    print("🚶 Starting Agent Control Demo...")
    from demo_agent_control import demo_random_agent
    
    demo_random_agent()

def run_trained_agent():
    """Run trained agent demonstration"""
    import os
    model_path = 'models/dqn_model_final.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Trained model not found at {model_path}")
        print("Please train the agent first using:")
        print("  python train_agent.py --episodes 50")
        return 1
    
    print("=" * 60)
    print("🤖 Starting Trained Agent Demo...")
    print("=" * 60)
    print(f"Loading model from: {model_path}")
    print("A PyBullet window should open showing the simulation.")
    print("Watch the trained agent (pedestrian) cross the road!")
    print("Press Ctrl+C in the terminal to stop.")
    print("=" * 60)
    print()
    
    try:
        from evaluate_agent import evaluate_agent
        
        evaluate_agent(
            model_path=model_path,
            num_episodes=10,
            gui=True,
            max_steps=1000
        )
        return 0
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Error running trained agent: {e}")
        import traceback
        traceback.print_exc()
        return 1

def run_tests():
    """Run environment tests"""
    print("🧪 Running Environment Tests...")
    from test_environment import main as test_main
    return test_main()

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Crossroad RL Environment Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py basic           # Run basic simulation
  python launcher.py rl-demo         # Run RL demonstration
  python launcher.py agent-demo      # Run agent control demo (see pedestrian move)
  python launcher.py trained-agent   # Run trained agent (requires trained model)
  python launcher.py test            # Run tests
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['basic', 'rl-demo', 'test', 'agent-demo', 'trained-agent'],
        help='Simulation mode to run'
    )
    
    if len(sys.argv) == 1:
        # No arguments provided, show help and offer interactive mode
        parser.print_help()
        print("\n" + "="*50)
        print("Interactive Mode")
        print("="*50)
        print("1. Basic Simulation")
        print("2. RL Demo")
        print("3. Agent Control Demo (see pedestrian move)")
        print("4. Trained Agent Demo (see trained model in action)")
        print("5. Run Tests")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\nSelect mode (1-6): ").strip()
                if choice == '1':
                    run_basic_simulation()
                    break
                elif choice == '2':
                    run_rl_demo()
                    break
                elif choice == '3':
                    run_agent_demo()
                    break
                elif choice == '4':
                    return run_trained_agent()
                elif choice == '5':
                    return run_tests()
                elif choice == '6':
                    print("Goodbye! 👋")
                    return 0
                else:
                    print("Invalid choice. Please select 1-6.")
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                return 0
            except EOFError:
                print("\nGoodbye! 👋")
                return 0
    
    else:
        # Parse command line arguments
        args = parser.parse_args()
        
        if args.mode == 'basic':
            run_basic_simulation()
        elif args.mode == 'rl-demo':
            run_rl_demo()
        elif args.mode == 'agent-demo':
            run_agent_demo()
        elif args.mode == 'trained-agent':
            return run_trained_agent()
        elif args.mode == 'test':
            return run_tests()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())