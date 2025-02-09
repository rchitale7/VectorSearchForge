import cdk = require("aws-cdk-lib");
import {GpuWorkerStack, SERVICE_PREFIX} from "./gpu_worker_stack";
import {CoordinatorStack} from "./coordinator_stack";

const SECURITY_GROUP_NAME: string = "security-group";
const TASK_ROLE_NAME: string = "taskRole";
const EXECUTION_ROLE_NAME: string = "executionRole";
const VPC_NAME: string = "vpc";

export interface CommonStackProps extends cdk.StackProps {
    readonly vpc: string,
    readonly securityGroup: string,
    readonly taskRole: string,
    readonly executionRole: string,
}

const app = new cdk.App();

const contextKey = app.node.tryGetContext('contextKey');
if (contextKey) {
    const nestedContext = app.node.tryGetContext(contextKey);
    if (nestedContext && typeof nestedContext === 'object') {
        Object.entries(nestedContext).forEach(([nestedKey, nestedValue]) => {
            app.node.setContext(nestedKey, nestedValue);
        });
    }
}

const region = app.node.tryGetContext('region') ?? process.env.CDK_DEFAULT_REGION;
const account = app.node.tryGetContext('account') ?? process.env.CDK_DEFAULT_ACCOUNT;
const prefix = `${app.node.tryGetContext('prefix')}` ?? SERVICE_PREFIX;
const vpc = `${app.node.tryGetContext(VPC_NAME)}`;
if (!vpc) {
    throw new Error("VPC must be defined");
}
const securityGroup = `${app.node.tryGetContext(SECURITY_GROUP_NAME)}`;

const taskRole = `${app.node.tryGetContext(TASK_ROLE_NAME)}`;
const executionRole = `${app.node.tryGetContext(EXECUTION_ROLE_NAME)}`;

const coordinatorStack = new CoordinatorStack(app, `${prefix}Coordinator`, {
    env: {
        region: region,
        account: account
    },
    vpc: vpc,
    securityGroup: securityGroup,
    taskRole: taskRole,
    executionRole: executionRole
});

const workerStack = new GpuWorkerStack(app, `${prefix}WorkerTask`, {
    env: {
        region: region,
        account: account
    },
    // Passing load balancer to Worker Fleet
    loadBalancer: coordinatorStack.getLoadBalancer(),
    vpc: vpc,
    securityGroup: securityGroup,
    taskRole: taskRole,
    executionRole: executionRole
});

// Coordinator should be spun up first so that when workers come up they are registered automatically
workerStack.addDependency(coordinatorStack);

app.synth();