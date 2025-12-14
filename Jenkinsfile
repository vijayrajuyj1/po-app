pipeline {
    agent any

    tools {
        sonarRunner 'sonar-scanner'
    }

    environment {
        AWS_REGION     = 'us-east-1'
        ECR_REPO_1     = '181975986508.dkr.ecr.us-east-1.amazonaws.com/po_conditioning_v1'
        DOCKER_IMAGE_1 = "${ECR_REPO_1}:${BUILD_NUMBER}"

        SONAR_URL      = 'http://3.235.193.244:9000'

        GIT_REPO_NAME  = 'po-app'
        GIT_USER_NAME  = 'vijayarajuyj1'
    }

    stages {

        stage('Static Code Analysis - Python') {
            steps {
                echo 'Performing static code analysis for Python with SonarQube...'

                withCredentials([
                    string(credentialsId: 'sonarqube', variable: 'SONAR_AUTH_TOKEN')
                ]) {
                    sh '''
                        sonar-scanner \
                          -Dsonar.projectKey=python-backend \
                          -Dsonar.projectName=python-backend \
                          -Dsonar.sources=. \
                          -Dsonar.python.version=3.10 \
                          -Dsonar.sourceEncoding=UTF-8 \
                          -Dsonar.host.url=${SONAR_URL} \
                          -Dsonar.login=${SONAR_AUTH_TOKEN}
                    '''
                }
            }
        }

        stage('Login to ECR') {
            steps {
                withCredentials([
                    [$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-cred']
                ]) {
                    sh '''
                        aws ecr get-login-password --region ${AWS_REGION} \
                        | docker login --username AWS --password-stdin ${ECR_REPO_1}
                    '''
                }
            }
        }

        stage('Build and Push Docker Image') {
            steps {
                sh '''
                    docker build -t ${DOCKER_IMAGE_1} .
                    docker push ${DOCKER_IMAGE_1}
                '''
            }
        }

        stage('Update Helm values.yaml and Push') {
            steps {
                withCredentials([
                    string(credentialsId: 'github', variable: 'GITHUB_TOKEN')
                ]) {
                    sh '''
                        git config user.email "vijayarajuyj1@gmail.com"
                        git config user.name "vijayarajuyj1"

                        sed -i "s/tag:.*/tag: ${BUILD_NUMBER}/" po-app1/values.yaml

                        git add po-app1/values.yaml
                        git commit -m "Update image tag to ${BUILD_NUMBER}" || echo "No changes to commit"
                        git push https://${GITHUB_TOKEN}@github.com/${GIT_USER_NAME}/${GIT_REPO_NAME}.git HEAD:main
                    '''
                }
            }
        }
    }
}
