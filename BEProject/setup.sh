# setup.sh
#!/bin/bash

set -e  # Exit on error

echo "=================================================="
echo "Dysarthric Speech Conversion - Complete Setup"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "\n${YELLOW}[1/10] Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi
python3 --version
echo -e "${GREEN}✓ Python OK${NC}"

# Step 2: Create directories
echo -e "\n${YELLOW}[2/10] Creating directory structure...${NC}"
mkdir -p data/raw/0 data/raw/1
mkdir -p checkpoints logs cache outputs
mkdir -p test_data pretrained
echo -e "${GREEN}✓ Directories created${NC}"

# Step 3: Create virtual environment
echo -e "\n${YELLOW}[3/10] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Step 4: Activate and install backend dependencies
echo -e "\n${YELLOW}[4/10] Installing backend dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Step 5: Check Node.js
echo -e "\n${YELLOW}[5/10] Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi
node --version
echo -e "${GREEN}✓ Node.js OK${NC}"

# Step 6: Install frontend dependencies
echo -e "\n${YELLOW}[6/10] Installing frontend dependencies...${NC}"
cd frontend
npm install
cd ..
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"

# Step 7: Create .env file if not exists
echo -e "\n${YELLOW}[7/10] Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Data Paths
DATA_ROOT=./data
CHECKPOINT_DIR=./checkpoints
LOG_DIR=./logs

# Training
BATCH_SIZE=4
NUM_EPOCHS=500
LEARNING_RATE=0.0002

# Optimization
USE_QUANTIZATION=false
USE_HALF_PRECISION=false

# Frontend
REACT_APP_SERVER_URL=ws://localhost:8000
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Step 8: Generate test data
echo -e "\n${YELLOW}[8/10] Generating test data...${NC}"
python3 scripts/generate_test_data.py --num-files 10 --output-dir data/raw
echo -e "${GREEN}✓ Test data generated${NC}"

# Step 9: Run tests
echo -e "\n${YELLOW}[9/10] Running system tests...${NC}"
python3 scripts/test_local.py
echo -e "${GREEN}✓ Tests completed${NC}"

# Step 10: Summary
echo -e "\n${YELLOW}[10/10] Setup Summary${NC}"
echo
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
echo ""
echo "Your system is ready to use!"
echo ""
echo "Next steps:" echo "  1. Train model:    python scripts/train.py --epochs 10" echo "  2. Start backend:  python -m backend.app.main"
echo "  3. Start frontend: cd frontend && npm start"
echo ""
echo "Or use Docker:      docker-compose up --build"