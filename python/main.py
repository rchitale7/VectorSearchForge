import getopt
import sys
from python.workload.workload import runWorkload
from python.data_types.data_types import IndexTypes, WorkloadTypes


def main(argv):
    opts, args = getopt.getopt(argv, "", ["workload=", "index_type=", "workload_type=", "h"])
    workloadName = None
    indexType = None
    workloadType = WorkloadTypes.INDEX_AND_SEARCH
    for opt, arg in opts:
        if opt == '--h':
            print('--dataset_file <dataset file path>')
            print(f'--index_type should have a value {IndexTypes.list()}')
            print(f'--workload_type(optional) should have a value {WorkloadTypes.list()} default is : {WorkloadTypes.INDEX_AND_SEARCH.value}')
            sys.exit()
        elif opt in "--workload":
            workloadName = arg
        elif opt == '--index_type':
            indexType = IndexTypes.from_str(arg)
        elif opt == "--workload_type":
            workloadType = WorkloadTypes.from_str(arg)

    runWorkload(workloadName, indexType, workloadType)


if __name__ == "__main__":
    main(sys.argv[1:])
