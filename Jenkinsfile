pipeline {
    agent any

    environment {
        VENV_DIR = '.venv'
        PYTHON_EXE = 'C:/Users/omkar/AppData/Local/Programs/Python/Python314/python.exe'
    }

    stages {

        stage('Set up Python environment') {
            steps {
                bat '''
                    "%PYTHON_EXE%" -m venv %VENV_DIR%
                    call %VENV_DIR%/Scripts/activate.bat
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Pull dataset with DVC') {
            steps {
                bat '''
                    call %VENV_DIR%/Scripts/activate.bat
                    dvc pull
                '''
            }
        }

        stage('Reproduce DVC pipeline') {
            steps {
                bat '''
                    call %VENV_DIR%/Scripts/activate.bat
                    dvc repro
                '''
            }
        }

        stage('Enforce recall gate') {
            steps {
                bat '''
                    call %VENV_DIR%/Scripts/activate.bat
                    python src/evaluate.py --min-recall 0.45
                '''
            }
        }

        stage('Archive model artifacts') {
            steps {
                archiveArtifacts artifacts: 'models/best_model.pkl, models/preprocessor.pkl',
                    fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully -- model trained, evaluated, and artifacts archived.'
        }

        failure {
            echo 'Pipeline failed -- check the stage logs above.'
        }
    }
}
