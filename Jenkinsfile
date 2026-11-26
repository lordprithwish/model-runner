pipeline {
    agent any

    options {
        disableConcurrentBuilds()
    }

    environment {
        REGISTRY   = 'ghcr.io/lordprithwish/model-runner'
        // The host needs an NVIDIA GPU, its driver and the NVIDIA Container Toolkit.
        HOST       = 'deploy@139.84.210.126'
        DEPLOY_DIR = '/opt/servers/model-runner'
    }

    stages {
        stage('Set Image Tag') {
            steps {
                script {
                    env.TAG = sh(
                        script: 'git rev-parse --short=12 HEAD',
                        returnStdout: true
                    ).trim()
                }

                echo "Building Model Runner commit: ${env.TAG}"
            }
        }

        stage('Verify SSH') {
            steps {
                sshagent(['deploy-production']) {
                    sh '''
                        ssh -o BatchMode=yes \
                            -o StrictHostKeyChecking=accept-new \
                            "$HOST" \
                            'whoami && hostname'
                    '''
                }
            }
        }

        stage('Build') {
            steps {
                sh 'docker compose build server'
            }
        }

        stage('Test') {
            steps {
                // tests/ isn't in the image (.dockerignore); pipe it in rather than bind-mount,
                // which breaks when the agent itself runs in a container. No GPU needed.
                sh '''
                    tar -c tests | docker run --rm -i \
                        "$REGISTRY/server:$TAG" \
                        sh -c 'tar -x && uv run --no-sync --with pytest --with httpx pytest -q -p no:cacheprovider'
                '''
            }
        }

        stage('Push Images') {
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'ghcr-publisher',
                        usernameVariable: 'REG_USER',
                        passwordVariable: 'REG_PASS'
                    )
                ]) {
                    sh '''
                        set +x

                        echo "$REG_PASS" | docker login ghcr.io \
                            -u "$REG_USER" --password-stdin

                        trap 'docker logout ghcr.io >/dev/null 2>&1 || true' EXIT

                        docker compose push server
                    '''
                }
            }
        }

        stage('Deploy') {
            steps {
                sshagent(['deploy-production']) {
                    withCredentials([
                        file(
                            credentialsId: 'model-runner-env',
                            variable: 'PROD_ENV_FILE'
                        ),
                        usernamePassword(
                            credentialsId: 'ghcr-production-reader',
                            usernameVariable: 'REG_USER',
                            passwordVariable: 'REG_PASS'
                        )
                    ]) {
                        sh '''
                            set +x

                            SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
                            SCP="scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

                            # Prepare production directory.
                            $SSH "$HOST" \
                                "mkdir -p '$DEPLOY_DIR'"

                            # Transfer Compose configuration.
                            $SCP compose.yaml \
                                "$HOST:$DEPLOY_DIR/compose.yaml"

                            # Generate production .env.
                            {
                                sed \
                                    -e '/^[[:space:]]*REGISTRY=/d' \
                                    -e '/^[[:space:]]*TAG=/d' \
                                    "$PROD_ENV_FILE"

                                printf '\\nREGISTRY=%s\\nTAG=%s\\n' \
                                    "$REGISTRY" "$TAG"
                            } | $SSH "$HOST" \
                                "umask 077 && \
                                cat > '$DEPLOY_DIR/.env.tmp' && \
                                mv '$DEPLOY_DIR/.env.tmp' '$DEPLOY_DIR/.env'"

                            # Authenticate production server to GHCR.
                            printf '%s' "$REG_PASS" | $SSH "$HOST" \
                                "docker login ghcr.io -u '$REG_USER' --password-stdin"

                            # Always attempt logout, even if deployment fails.
                            trap '$SSH "$HOST" "docker logout ghcr.io" >/dev/null 2>&1 || true' EXIT

                            # Pull and deploy the new release.
                            # Restarting drops queued/running jobs (in-memory queue); finished videos in outputs/ stay.
                            $SSH "$HOST" \
                                "cd '$DEPLOY_DIR' && \
                                docker compose pull && \
                                docker compose up -d \
                                    --no-build --remove-orphans"

                            # Image cleanup on production is NOT done per deploy. Old tagged releases are kept
                            # for rollback and removed by a weekly cron on the host (deploy user's crontab):
                            #   0 4 * * 0 docker image prune -af --filter "until=168h" --filter label=model-runner.app >/dev/null 2>&1
                            # These images are several GB each (torch + CUDA), so don't skip that cron.
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            // Every build leaves a uniquely tagged multi-GB image; keep three days of this project's only.
            // The label (set in compose.yaml) stops this touching other projects on this Docker daemon.
            sh 'docker image prune -af --filter until=72h --filter label=model-runner.app'
        }
    }
}
