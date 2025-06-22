import os
import subprocess
import sys

def create_virtual_env(venv_name="venv"):
    """创建虚拟环境并安装依赖"""
    try:
        # 1. 创建虚拟环境
        print(f"Creating virtual environment '{venv_name}'...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])

        # 2. 安装依赖
        print("Installing dependencies...")
        pip_path = os.path.join(venv_name, "Scripts" if os.name == "nt" else "bin", "pip")
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])

        # 3. 生成激活脚本提示
        print("\n✅ Environment setup complete!")
        print(f"\nRun the following command to activate the environment:")
        if os.name == "nt":  # Windows
            print(f"  .\\{venv_name}\\Scripts\\activate")
        else:  # Linux/Mac
            print(f"  source {venv_name}/bin/activate")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_virtual_env()