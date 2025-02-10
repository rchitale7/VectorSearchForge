import autoscaling = require("aws-cdk-lib/aws-autoscaling");
import ec2 = require("aws-cdk-lib/aws-ec2");
import ecs = require("aws-cdk-lib/aws-ecs");
import cdk = require("aws-cdk-lib");
import {SecurityGroup} from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import {Size} from "aws-cdk-lib";
import {Protocol} from "aws-cdk-lib/aws-ecs";
import {ApplicationLoadBalancer} from "aws-cdk-lib/aws-elasticloadbalancingv2";
import {Construct} from "constructs";
import {CommonStackProps} from "./index";

export const COORDINATOR_IMAGE_NAME: string = "coordinator-image";

export class CoordinatorStack extends cdk.Stack {
    private coordinatorImage: string;
    private readonly loadBalancer: ApplicationLoadBalancer;

    private initClassParams(scope: cdk.App) {
        this.coordinatorImage = `${scope.node.tryGetContext(COORDINATOR_IMAGE_NAME)}`;
    }

    private getVPC(scope: Construct, id: string, props: CommonStackProps) {
        return ec2.Vpc.fromLookup(scope, `${id}VPC`, {
            vpcId: props.vpc
        });
    }

    constructor(scope: cdk.App, id: string, props: CommonStackProps) {
        super(scope, id, props);
        this.initClassParams(scope);
        const vpc = this.getVPC(this, id, props);

        // Autoscaling group that will launch a fleet of instances that have C5 instances
        const asg = new autoscaling.AutoScalingGroup(this, id + "-asg", {
            instanceType: ec2.InstanceType.of(
                ec2.InstanceClass.C5,
                ec2.InstanceSize.XLARGE4
            ),
            desiredCapacity: 1,
            machineImage: ec2.MachineImage.fromSsmParameter(
                '/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id'
            ),
            vpc,
            maxCapacity: 3,
            securityGroup: SecurityGroup.fromLookupById(this, id + "-sg", props.securityGroup)
        });

        // Attach the fleet to an ECS cluster with a capacity provider.
        // This capacity provider will automatically scale up the ASG
        // to launch more GPU instances when GPU tasks need them.
        const cluster = new ecs.Cluster(this, id + "EcsCluster", { vpc: vpc, containerInsights: true});
        const capacityProvider = new ecs.AsgCapacityProvider(
            this,
            id + "AsgCapacityProvider",
            {autoScalingGroup: asg}
        );
        cluster.addAsgCapacityProvider(capacityProvider);

        const coordinatorTaskDefinition = new ecs.Ec2TaskDefinition(this, id + "coordinator-task", {
            networkMode: ecs.NetworkMode.BRIDGE, // This will ensure that we are able to access the IP of the machine
            family: "cdk-coordinator-task-definition",
            taskRole: iam.Role.fromRoleArn(this, id + "coordinator-task-role", props.taskRole),
            executionRole: iam.Role.fromRoleArn(this, id + "coordinator-task-execution", props.executionRole),
            volumes: [
                {
                    "name": "coordinator-volume",
                    "host": {}
                }
            ]
        });


        coordinatorTaskDefinition.addContainer( id + "-coordinator", {
            containerName:  "coordinator-container",
            essential: true,
            image: new ecs.RepositoryImage(this.coordinatorImage),

            cpu: 10 * 1024,
            memoryLimitMiB: 30 * 1024, // hard limit, max we have is 32
            memoryReservationMiB: 30 * 1024, // soft limit
            portMappings: [
                {
                    containerPort: 6006,
                    hostPort: 6006,
                    protocol: Protocol.TCP
                }
            ],
            entryPoint: [
                "python",
                "app.py"
            ],
            environment: {
                "DOMAIN": "prod",
                "PYTHONFAULTHANDLER": "1",
                "PYTHONUNBUFFERED": "1"
            },
            workingDirectory: "/app",
            user: "appuser",
            ulimits: [
                {
                    name: ecs.UlimitName.CORE,
                    softLimit: -1,
                    hardLimit: -1,
                },
            ],
            startTimeout:  cdk.Duration.seconds(300),
            logging: new ecs.AwsLogDriver({
                streamPrefix: "ecs",
                mode: ecs.AwsLogDriverMode.NON_BLOCKING,
                maxBufferSize: Size.mebibytes(25),
                logGroup: new cdk.aws_logs.LogGroup(this, id + "coordinator-service-loggroup", {
                    logGroupName: "/ecs/cdk-coordinator-task-definition",
                    removalPolicy: cdk.RemovalPolicy.DESTROY,
                })

            })
        }).addMountPoints({
            containerPath: "/app/logs",
            readOnly: false,
            sourceVolume: "coordinator-volume"
        });

        // Request ECS to launch the task onto the fleet
        const coordinatorService = new ecs.Ec2Service(this, id + "coordinator-service", {
            cluster,
            desiredCount: 1,
            // Service will automatically request capacity from the
            // capacity provider
            capacityProviderStrategies: [
                {
                    capacityProvider: capacityProvider.capacityProviderName,
                    base: 0,
                    weight: 1,
                },
            ],
            taskDefinition: coordinatorTaskDefinition,
        });

        this.loadBalancer = new ApplicationLoadBalancer(this, id + 'ALB', {
            vpc,
            internetFacing: false,
            loadBalancerName: id + 'ALB',
            securityGroup: SecurityGroup.fromSecurityGroupId(this, `${id}1-sg`, props.securityGroup)
        });

        const listener = this.loadBalancer.addListener(id + 'ALBListener', {
            port: 80,
            open: true,
        });

        listener.addTargets(id + 'ALBTarget', {
            port: 80,
            targets: [coordinatorService],
            healthCheck: {
                enabled: true,
                healthyThresholdCount: 2,
                unhealthyThresholdCount: 2,
                path: '/'
            }
        })
    }

    getLoadBalancer(): ApplicationLoadBalancer {
        return this.loadBalancer;
    }
}