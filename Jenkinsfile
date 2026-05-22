pipeline {
    agent any

    triggers {
        githubPush()
    }

    environment {
        PATH = "/opt/homebrew/bin:/usr/local/bin:${env.PATH}"
        DOCKER_HOST = "unix://${env.HOME}/.colima/default/docker.sock"
        DOCKER_REGISTRY = 'jman003'
        APP_NAME_BACKEND = 'lawracle-backend'
        APP_NAME_FRONTEND = 'lawracle-frontend'
        IMAGE_TAG = "${env.BUILD_ID}"
        DOCKER_CONFIG = "${WORKSPACE}/.docker"
        DOCKER_BUILDKIT = '1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Backend Image') {
            steps {
                script {
                    dir('backend') {
                        withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                            sh """
                            echo "\$DOCKER_PASS" | docker login -u "\$DOCKER_USER" --password-stdin
                            docker pull ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:latest || true
                            docker build --cache-from ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:latest -t ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:${IMAGE_TAG} .
                            """
                        }
                    }
                }
            }
        }

        stage('Test Backend') {
            steps {
                script {
                    dir('backend') {
                        sh """
                        docker run --rm -v \$(pwd)/tests:/app/tests -v \$(pwd)/requirements-test.txt:/app/requirements-test.txt ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:${IMAGE_TAG} sh -c "pip3 install --no-cache-dir -r requirements-test.txt && PYTHONPATH=/app pytest tests/"
                        """
                    }
                }
            }
        }

        stage('Push Backend Image') {
            steps {
                script {
                    dir('backend') {
                        sh """
                        docker push ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:${IMAGE_TAG}
                        docker tag ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:latest
                        docker push ${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:latest
                        """
                    }
                }
            }
        }

        stage('Build & Push Frontend Image') {
            steps {
                script {
                    dir('frontend') {
                        withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                            sh """
                            echo "\$DOCKER_PASS" | docker login -u "\$DOCKER_USER" --password-stdin
                            docker pull ${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:latest || true
                            docker build --cache-from ${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:latest -t ${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:${IMAGE_TAG} .
                            docker push ${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:${IMAGE_TAG}
                            docker tag ${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:latest
                            docker push ${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:latest
                            """
                        }
                    }
                }
            }
        }

        stage('Deploy via Ansible') {
            steps {
                script {
                    dir('ansible') {
                        sh """
                            ansible-playbook -i inventory.ini deploy.yml \\
                                -e backend_image="${DOCKER_REGISTRY}/${APP_NAME_BACKEND}:${IMAGE_TAG}" \\
                                -e frontend_image="${DOCKER_REGISTRY}/${APP_NAME_FRONTEND}:${IMAGE_TAG}"
                        """
                    }
                }
            }
        }
    }
}
