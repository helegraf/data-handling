import java.sql.SQLException;

import dataHandling.mySQL.MetaDataDataBaseConnection;

public class dataHandlingTest {
	public static void main (String [] args) throws SQLException {
		MetaDataDataBaseConnection databaseConn = new MetaDataDataBaseConnection(args[0], args[1], args[2], args[3]);
		
		System.out.println("Test available sets");
		System.out.println(databaseConn.getAvailableClassifierSets());
		System.out.println(databaseConn.getAvailableDataSetSets());
		System.out.println(databaseConn.getAvailableMetaDataSets());
		
		String datasetOrigin = "openML_dataset_id";
		String classifierSetName = "all-standard_config";
		String dataSetSetName="all";
		String setName = classifierSetName;
		String metaDataSetName="all";
		int datasetId = 3;
		
		System.out.println("Test available classifier data");
		System.out.println(databaseConn.getClassifierPerformancesForDataSet(3, datasetOrigin, classifierSetName));
		System.out.println(databaseConn.getClassifierPerformancesForDataSetSet(dataSetSetName, classifierSetName));
		
		System.out.println("Test available members of data sets");
		System.out.println(databaseConn.getMembersOfClassifierSet(setName));
		setName=dataSetSetName;
		System.out.println(databaseConn.getMembersOfDatasetSet(setName));
		setName=metaDataSetName;
		System.out.println(databaseConn.getMembersOfMetadataSet(setName));
		
		System.out.println("Test available metafeature data");
		System.out.println(databaseConn.getMetaDataSetForDataSet(datasetId, datasetOrigin, metaDataSetName));
		System.out.println(databaseConn.getMetaDataSetForDataSetSet(dataSetSetName, metaDataSetName));
	}
}
