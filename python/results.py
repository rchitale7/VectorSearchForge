import csv
import getopt
import json
import logging
import sys
from python.data_types.data_types import IndexTypes, WorkloadTypes
from python.utils.common_utils import ensureDir, formatTimingMetricsValue


def persistMetricsAsCSV(workloadType: WorkloadTypes, allMetrics: dict, workloadName: str, indexType: IndexTypes):
    ensureDir(f"results/{workloadName}")
    fields = ["workload-name", "indexType", "dataset-name", "dimensions", "vectors-count", "queries-count", "indexing-params", "index-creation-time", "write-to-file-time", "total-build-time", "search-parameter", "search-time", "unit", "recall@100", "recall@1"]
    rows = []
    if workloadType == WorkloadTypes.INDEX:
        logging.error("This type of workload is not supported for writing data in csv")
        sys.exit()
    else:
        workloadDetails = allMetrics[workloadName]["workload-details"]
        searchParamItr = 0
        indexingParamItr = 0
        for searchMetric in allMetrics[workloadName]["searchMetrics"]:
            searchParamItr = searchParamItr + 1
            searchTimingMetrics = searchMetric["search-timing-metrics"]
            row = {
                "workload-name": workloadName,
                "indexType": indexType.value,
                "dataset-name": workloadDetails["dataset_name"],
                "dimensions": workloadDetails["dimension"],
                "queries-count": workloadDetails.get("queriesCount"),
                "vectors-count": workloadDetails.get("vectorsCount"),
                "indexing-params": searchMetric["indexing-params"],
                "search-time": formatTimingMetricsValue(searchTimingMetrics['searchTime']),
                "unit": searchTimingMetrics["units"],
                "recall@100": searchTimingMetrics["recall_at_100"],
                "recall@1": searchTimingMetrics["recall_at_1"],
                "search-parameter": searchMetric["search-params"],
            }

            if allMetrics[workloadName].get("indexingMetrics") is not None:
                row["vectors-count"] = workloadDetails["vectorsCount"]
                indexingMetrics = allMetrics[workloadName]["indexingMetrics"]
                #float(f"{searchTimingMetrics['searchTime']:.4f}"),
                row["index-creation-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"]["indexTime"])
                row["write-to-file-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"]["writeIndexTime"])
                row["total-build-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"]["totalTime"])
                if searchParamItr % len(workloadDetails["search-parameters"]) == 0:
                    indexingParamItr = indexingParamItr + 1
            rows.append(row)
            

    with open(f"results/{workloadName}/{workloadType.value}_{indexType.value}.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        # writing headers (field names)
        writer.writeheader()
        # writing data rows
        writer.writerows(rows)
    
    logging.info(f"Results are stored at location: results/{workloadName}/{workloadType.value}_{indexType.value}.csv")


def writeDataInCSV(workloadName:str, indexType:IndexTypes, workloadType:WorkloadTypes):
    if workloadType == WorkloadTypes.INDEX:
        logging.error("This type of workload is not supported for writing data in csv")
        sys.exit()
    jsonFile = f"results/{workloadName}/{workloadType.value}_{indexType.value}.json"
    f = open(jsonFile)
    allMetrics = json.load(f)
    f.close
    persistMetricsAsCSV(workloadType, allMetrics, workloadName, indexType)
    

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

    writeDataInCSV(workloadName, indexType, workloadType)

if __name__ == "__main__":
    main(sys.argv[1:])