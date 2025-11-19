# 🔧 Troubleshooting Guide

Having issues? This guide covers common problems and their solutions, organized from most common to least common.

## 🚨 Quick Diagnosis

### Check These First:
1. **Python version**: `python --version` (need 3.8+)
2. **Dependencies installed**: `pip list` (check for pybullet, torch, etc.)
3. **File permissions**: Make sure you can read/write in project directory
4. **Available memory**: Close other applications if low on RAM

## 🐛 Common Installation Issues

### Problem: "No module named 'pybullet'"
```bash
ModuleNotFoundError: No module named 'pybullet'
```

**Solution**:
```bash
# Try reinstalling with latest pip
pip install --upgrade pip
pip install pybullet --upgrade

# If still fails, try with conda
conda install pybullet -c conda-forge

# Or install specific version
pip install pybullet==3.2.5
```

### Problem: PyBullet won't start/crashes immediately
```bash
ImportError: Failed to import pybullet
```

**Solutions**:

**On Linux:**
```bash
# Install OpenGL libraries
sudo apt-get install python3-opengl
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3

# Install virtual display (for headless systems)
sudo apt-get install xvfb
```

**On macOS:**
```bash
# Update to latest macOS
# Install Xcode command line tools
xcode-select --install
```

**On Windows:**
```bash
# Update Visual C++ Redistributables
# Download from Microsoft website
# Restart computer after installation
```

### Problem: "No module named 'torch'"
```bash
ModuleNotFoundError: No module named 'torch'
```

**Solution**:
```bash
# Install PyTorch (visit pytorch.org for latest)
pip install torch torchvision torchaudio

# For specific systems:
# CPU only: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Permission Denied errors
```bash
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
```bash
# Option 1: Install in user directory
pip install --user -r requirements.txt

# Option 2: Use virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt

# Option 3: Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER /path/to/project
```

## 🖥️ Display and Graphics Issues

### Problem: PyBullet window is black/empty
```
PyBullet window opens but shows nothing
```

**Solutions**:
1. **Update graphics drivers** (most common fix)
2. **Try different connection mode**:
   ```python
   # In intersection_env.py, try:
   p.connect(p.DIRECT)  # No graphics
   # or 
   p.connect(p.GUI_SERVER)  # Alternative graphics mode
   ```
3. **Disable shadows**:
   ```python
   # In config.py:
   SIMULATION_CONFIG['enable_shadows'] = False
   ```

### Problem: "Unable to init server display"
```bash
pybullet.error.Error: Unable to init server display
```

**Linux Headless Systems**:
```bash
# Install and use virtual display
sudo apt-get install xvfb
xvfb-run -a python launcher.py test

# Or run without GUI
python train_agent.py --no-gui
```

### Problem: Low FPS/choppy graphics
```
Simulation runs very slowly
```

**Solutions**:
1. **Reduce quality in config.py**:
   ```python
   SIMULATION_CONFIG = {
       'fps': 30,                    # Lower target FPS  
       'enable_shadows': False,      # Disable shadows
       'physics_solver_iterations': 20,  # Reduce physics quality
   }
   ```

2. **Close other applications** 
3. **Train without GUI**: `python launcher.py train` → choose 'n' for GUI
4. **Reduce number of cars**:
   ```python
   CAR_CONFIG['num_cars'] = 4  # Instead of 8
   ```

## 🎓 Training Issues

### Problem: Training never improves (scores stay negative)
```
Episode 200+ but average score still -50 or worse
```

**Diagnosis Steps**:
1. **Check if agent is moving**:
   ```bash
   # Train with GUI to see what's happening
   python launcher.py train
   # Choose 'y' for GUI and watch agent
   ```

2. **Check reward signals**:
   - Are waypoints reachable?
   - Is collision penalty too harsh?
   - Are there progress rewards?

**Solutions**:
```python
# In config.py, try easier settings:
RL_CONFIG['reward_structure']['collision_penalty'] = -50  # Less harsh
RL_CONFIG['reward_structure']['progress_reward'] = 5      # More progress reward
PEDESTRIAN_CONFIG['waypoint_tolerance'] = 1.2            # Easier targets

# Or reduce environment difficulty:
CAR_CONFIG['num_cars'] = 4                               # Fewer cars
CAR_CONFIG['rule_breaker_probability'] = 0.1             # More predictable
```

### Problem: Training crashes with CUDA/GPU errors
```bash
RuntimeError: CUDA out of memory
```

**Solutions**:
```python
# Force CPU usage in dqn_agent.py:
device = torch.device("cpu")

# Or reduce batch size in config.py:
RL_CONFIG['batch_size'] = 32  # Instead of 128
RL_CONFIG['replay_buffer_size'] = 10000  # Instead of 50000
```

### Problem: Agent gets stuck in corners/walls
```
Agent stops moving or moves in circles
```

**Solutions**:
1. **Check boundary constraints** in `gym_crossroad_env.py`:
   ```python
   # Make sure boundaries aren't too restrictive
   new_pos[0] = np.clip(new_pos[0], -8.0, 8.0)  # Wider area
   new_pos[1] = np.clip(new_pos[1], -8.0, 8.0)
   ```

2. **Add movement encouragement**:
   ```python
   # In reward function, penalize staying in same place
   if action == 0:  # Stay action
       reward -= 0.2  # Stronger penalty for not moving
   ```

### Problem: Training too slow
```
Taking hours to train 500 episodes
```

**Speed Optimizations**:
1. **Disable GUI**: Never train with graphics for speed
2. **Reduce episode length**: 
   ```python
   RL_CONFIG['max_episode_steps'] = 1000  # Instead of 2000
   ```
3. **Optimize physics**:
   ```python
   SIMULATION_CONFIG['physics_timestep'] = 1/120  # Instead of 1/240
   ```
4. **Batch training**:
   ```python
   RL_CONFIG['train_frequency'] = 8  # Train every 8 steps instead of 4
   ```

## 💾 Model Loading/Saving Issues

### Problem: "Model file not found"
```bash
FileNotFoundError: models/dqn_model_final.pth not found
```

**Solutions**:
1. **Train a model first**:
   ```bash
   python launcher.py train
   ```
2. **Check model directory**:
   ```bash
   ls models/  # See what models exist
   ```
3. **Use specific model**:
   ```bash
   python evaluate_agent.py --model models/dqn_model_episode_100.pth
   ```

### Problem: Model loads but performs poorly
```
Model loads successfully but agent behaves randomly
```

**Possible Causes**:
1. **Model undertrained**: Train for more episodes
2. **Wrong model version**: Check model timestamp
3. **Configuration mismatch**: Model trained with different settings

**Solutions**:
```bash
# Train longer
python train_agent.py --episodes 1000

# Check multiple saved models
python evaluate_agent.py --model models/dqn_model_episode_500.pth
```

## 🚗 Simulation Behavior Issues

### Problem: Cars drive through each other
```
Vehicles overlap and pass through walls/other cars
```

**Solutions**:
1. **Check physics timestep**:
   ```python
   SIMULATION_CONFIG['physics_timestep'] = 1/240  # Smaller = more accurate
   ```
2. **Increase solver iterations**:
   ```python
   SIMULATION_CONFIG['physics_solver_iterations'] = 100
   ```

### Problem: Traffic lights don't work properly
```
All lights show same color or don't change
```

**Debug Steps**:
1. **Check console output** for signal states
2. **Verify timing** in config:
   ```python
   ENVIRONMENT_CONFIG['signal_change_interval'] = 300  # 5 seconds
   ```

### Problem: Cars all moving in same direction
```
No cars coming from some directions
```

**Solution**:
Check car spawn positions in `intersection_env.py` spawn_cars() function.

## 🖱️ User Interface Issues

### Problem: Launcher menu not working
```
Interactive menu doesn't respond to input
```

**Solutions**:
```bash
# Try direct commands instead:
python launcher.py test
python launcher.py train  
python launcher.py run

# Check for input encoding issues:
export LC_ALL=C.UTF-8  # Linux/macOS
```

### Problem: Keyboard interrupt doesn't work
```
Ctrl+C doesn't stop training
```

**Solutions**:
- **Wait a few seconds** for graceful shutdown
- **Force quit**: Ctrl+Z then `kill %1`
- **Close terminal window** as last resort

## 🧠 Memory Issues

### Problem: "Memory Error" or system freezes
```bash
MemoryError: Unable to allocate memory
```

**Solutions**:
1. **Reduce replay buffer size**:
   ```python
   RL_CONFIG['replay_buffer_size'] = 10000  # Instead of 50000
   ```
2. **Smaller batch size**:
   ```python
   RL_CONFIG['batch_size'] = 32  # Instead of 128
   ```
3. **Fewer cars**:
   ```python
   CAR_CONFIG['num_cars'] = 4  # Instead of 8
   ```
4. **Close other applications**
5. **Use CPU instead of GPU** for training

## 🔍 Debugging Tools

### Enable Debug Output:
```python
# Add to top of launcher.py:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Resources:
```bash
# Linux/macOS:
htop  # Monitor CPU/memory usage

# Windows:
Task Manager  # Monitor resource usage
```

### Test Components Individually:
```bash
# Test environment only:
python -c "from intersection_env import CrossroadEnvironment; env = CrossroadEnvironment(); env.run_simulation()"

# Test agent only:
python -c "from dqn_agent import DQNAgent; agent = DQNAgent(39, 5); print('Agent created')"
```

## 📞 Getting Additional Help

### Before Asking for Help:
1. **Check error messages carefully**
2. **Try solutions in this guide**
3. **Search for specific error messages online**
4. **Test with minimal example**

### Information to Include:
- **Operating system** (Windows/macOS/Linux + version)
- **Python version**: `python --version`  
- **Package versions**: `pip list`
- **Full error message** (copy-paste, don't summarize)
- **What you were trying to do**
- **What you've already tried**

### Quick Health Check Script:
```python
# save as test_setup.py and run
try:
    import pybullet as p
    print("✅ PyBullet imported successfully")
    
    import torch
    print(f"✅ PyTorch imported: {torch.__version__}")
    
    import gymnasium as gym
    print("✅ Gymnasium imported successfully")
    
    import numpy as np
    print("✅ NumPy imported successfully")
    
    # Test PyBullet connection
    p.connect(p.DIRECT)
    print("✅ PyBullet connection successful")
    p.disconnect()
    
    print("\n🎉 All systems working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check installation guide for solutions")
```

## ✅ Prevention Tips

### Keep Your Setup Healthy:
1. **Use virtual environments** for Python projects
2. **Keep dependencies updated**: `pip install --upgrade -r requirements.txt`
3. **Regular system updates** (OS, drivers, etc.)
4. **Monitor disk space** (models and logs take space)
5. **Backup working configurations**

**Still having issues? Try starting with the simplest possible setup and gradually add complexity.**