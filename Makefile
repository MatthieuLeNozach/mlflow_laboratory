# Define the path to your environment files
ENV_DIR := .environment

# Assuming you have .env and .env.secret files inside the .environment directory
ENV_FILES := .environment/.env .environment/.env.secret

# Ensure the environment files have read permissions
$(shell chmod +r $(ENV_FILES))

# Export all variables from the .env and .env.secret files
include $(ENV_FILES)
export $(shell sed 's/=.*//' $(ENV_FILES))

# Define the default target that runs when you just type 'make'
all: up

# Target for starting up the application with building images
up:
	@echo "Building and starting up services..."
	docker-compose up --build

# Target for shutting down the application
down:
	@echo "Shutting down services..."
	docker-compose down

# Target to view logs
logs:
	docker-compose logs

# Use this target to force a rebuild of your Docker containers
rebuild:
	@echo "Rebuilding and starting up services..."
	docker-compose up -d --build

# Add a help command to list available commands
help:
	@echo "Available commands:"
	@echo "  make up       - Build (if necessary) and start up services"
	@echo "  make down     - Shut down services"
	@echo "  make logs     - View logs"
	@echo "  make rebuild  - Force a rebuild and restart services"

.PHONY: all up down logs rebuild help
