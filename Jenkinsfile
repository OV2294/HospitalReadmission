pipeline {
    agent any

    environment {
        VENV_DIR = '.venv'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set up Python environment') {
            steps {
                sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Pull dataset with DVC') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    dvc pull || echo "No DVC remote configured yet — using data already in the workspace."
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    flake8 src/ api/ tests/ --max-line-length=100 --extend-ignore=E203,W503
                '''
            }
        }

        stage('Run unit tests') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pytest tests/ -v
                '''
            }
        }

        stage('Reproduce DVC pipeline') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    dvc repro
                '''
            }
        }

        stage('Enforce recall gate') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    python src/evaluate.py --min-recall 0.45
                '''
            }
        }

        stage('Archive model artifacts') {
            steps {
                archiveArtifacts artifacts: 'models/best_model.pkl, models/preprocessor.pkl', fingerprint: true
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build -t hospital-readmission-api .'
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully — model trained, evaluated, and image built.'
        }
        failure {
            echo 'Pipeline failed — check the stage logs above.'
        }
    }
}
