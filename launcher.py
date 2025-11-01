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
  python launcher.py test            # Run tests
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['basic', 'rl-demo', 'test'],
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
        print("3. Run Tests")
        print("4. Exit")
        
        while True:
            try:
                choice = input("\nSelect mode (1-4): ").strip()
                if choice == '1':
                    run_basic_simulation()
                    break
                elif choice == '2':
                    run_rl_demo()
                    break
                elif choice == '3':
                    return run_tests()
                elif choice == '4':
                    print("Goodbye! 👋")
                    return 0
                else:
                    print("Invalid choice. Please select 1-4.")
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
        elif args.mode == 'test':
            return run_tests()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())