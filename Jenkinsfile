pipeline {
    agent any

    environment {
        AWS_REGION     = 'us-east-1'
        ECR_REPO_1     = '181975986508.dkr.ecr.us-east-1.amazonaws.com/po_conditioning_v1'
        DOCKER_IMAGE_1 = "${ECR_REPO_1}:${BUILD_NUMBER}"

        GIT_REPO_NAME  = 'po-app'
        GIT_USER_NAME  = 'vijayrajuyj1'
    }

    stages {

        stage('Login to ECR') {
            steps {
                echo 'Logging into Amazon ECR...'
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
                echo 'Building and pushing Docker image...'
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
