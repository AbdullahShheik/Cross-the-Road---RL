# 📋 Project Setup Guide

This guide will help you set up the Crossroad RL project on your computer from scratch.

## 🎯 Prerequisites

Before starting, you'll need:

- **Python 3.8 or higher** - [Download from python.org](https://www.python.org/downloads/)
- **Git** (optional) - [Download from git-scm.com](https://git-scm.com/)
- **4GB+ RAM** - For running the 3D simulation smoothly
- **Graphics card** (recommended) - For better PyBullet performance

## 📥 Step 1: Download the Project

### Option A: Download ZIP
1. Go to the project repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract to your desired folder

### Option B: Clone with Git
```bash
git clone https://github.com/YourUsername/Cross-the-Road---RL.git
cd Cross-the-Road---RL
```

## 🐍 Step 2: Set Up Python Environment

### Check Python Version
```bash
python --version
# Should show Python 3.8 or higher
```

### Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv crossroad_env

# Activate it
# On Windows:
crossroad_env\Scripts\activate
# On macOS/Linux:
source crossroad_env/bin/activate
```

## 📦 Step 3: Install Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

### What Gets Installed:
- **pybullet** - 3D physics simulation
- **numpy** - Numerical computations  
- **gymnasium** - RL environment framework
- **torch** - Neural network training
- **matplotlib** - Plotting training progress

## ✅ Step 4: Test Installation

Run a quick test to make sure everything works:

```bash
python launcher.py test
```

You should see:
1. A PyBullet window opens showing the 3D crossroad
2. Cars driving around with traffic lights
3. Console output showing simulation status

**If you see the 3D simulation, congratulations! Setup is complete! 🎉**

## 🔧 Troubleshooting Installation

### Problem: "Command 'python' not found"
**Solution**: 
- Make sure Python is installed and added to PATH
- Try using `python3` instead of `python`
- On Windows, try `py` instead

### Problem: PyBullet won't install
**Solution**:
```bash
# Try updating pip first
pip install --upgrade pip

# Install PyBullet specifically
pip install pybullet --upgrade
```

### Problem: "No module named torch"
**Solution**:
```bash
# Install PyTorch with specific command for your system
# Visit: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
```

### Problem: Graphics/Display issues
**Solution**:
- Update your graphics drivers
- If on Linux, install: `sudo apt-get install xvfb`
- Try running in headless mode (without GUI) first

### Problem: Permission errors
**Solution**:
```bash
# Use --user flag to install in user directory
pip install --user -r requirements.txt
```

## 🖥️ System Requirements

### Minimum Requirements:
- **CPU**: Dual-core 2GHz
- **RAM**: 4GB 
- **Storage**: 1GB free space
- **OS**: Windows 10, macOS 10.14, or Ubuntu 18.04+

### Recommended:
- **CPU**: Quad-core 3GHz+
- **RAM**: 8GB+
- **GPU**: Dedicated graphics card
- **Storage**: SSD for faster loading

## 🚀 Next Steps

Once setup is complete:

1. **Test the simulation**: `python launcher.py test`
2. **Read the simulation guide**: [02_simulation.md](02_simulation.md)
3. **Start training**: `python launcher.py train`

## 💡 Pro Tips

- **Use virtual environments** to avoid package conflicts
- **Close other applications** during training for better performance
- **Update drivers** for optimal graphics performance
- **Start with small training episodes** (50-100) to test everything works

## 📞 Getting Help

If you encounter issues not covered here:

1. Check the [Troubleshooting Guide](06_troubleshooting.md)
2. Ensure all prerequisites are met
3. Try running each step individually
4. Check for typos in commands

**Ready for the next step? Let's explore the simulation! → [Understanding the Simulation](02_simulation.md)**