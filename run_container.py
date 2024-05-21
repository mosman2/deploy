import subprocess

def build_image(tag_name):
    """Build a Docker image from the Dockerfile in the current directory."""
    print("Building Docker image...")
    try:
        subprocess.check_call(['docker', 'build', '-t', tag_name, '.'])
        print(f"Docker image {tag_name} built successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to build the Docker image.")
        raise

def run_container(tag_name):
    """Run a Docker container from the built image."""
    print("Running Docker container...")
    try:
        subprocess.check_call(['docker', 'run', '--rm', '--name', 'script_container', tag_name])
        print("Docker container started successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to start the Docker container.")
        raise

def main():
    tag_name = 'my_deployable_script'
    build_image(tag_name)
    run_container(tag_name)

if __name__ == '__main__':
    main()
