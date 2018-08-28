

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

/**
 * Handles the saving and retrieving of classifier performance data for data
 * sets and meta features for data sets from a data base. Assumes a data base
 * containing the referenced tables has already been created.
 * 
 * @author Helena Graf
 *
 */
public class MetaDataDataBaseConnection {

	public static final String COLUMN_LABEL_METAFEATURE_COMPUTATION_TIME = "time";
	public static final String COLUMN_LABEL_METAFEATURE_RUN_ID = "metafeature_run_id";
	public static final String COLUMN_LABEL_METAFEATURE_VALUE = "metafeature_value";
	public static final String COLUMN_LABEL_METAFEATURE_SET_NAME = "metafeature_set_name";
	public static final String COLUMN_LABEL_METAFEATURE_GROUP_NAME = "metafeature_group";
	public static final String COLUMN_LABEL_METAFEATURE_NAME = "metafeature_name";

	public static final String TABLE_NAME_METAFEATURE_VALUES = "metafeature_values";
	public static final String TABLE_NAME_METAFEATURE_TIMES = "metafeature_times";
	public static final String TABLE_NAME_METAFEATURE_SET_MEMBERS = "metafeature_set_members";
	public static final String TABLE_NAME_METAFEATURE_SETS = "metafeature_sets";
	public static final String TABLE_NAME_METAFEATURE_GROUP_MEMBERS = "metafeature_group_members";
	public static final String TABLE_NAME_METAFEATURE_GROUPS = "metafeature_groups";
	public static final String TABLE_NAME_METAFEATURE_RUNS = "metafeature_runs";

	public static final String TABLE_NAME_DATASET_SETS = "dataset_sets";
	public static final String TABLE_NAME_DATASET_SET_MEMBERS = "dataset_set_members";

	public static final String COLUMN_LABEL_DATASET_ID = "dataset_id";
	public static final String COLUMN_LABEL_DATASET_NAME = "dataset_name";
	public static final String COLUMN_LABEL_DATASET_SET_NAME = "dataset_set_name";

	public static final String TABLE_NAME_CLASSIFIER_SETS = "classifier_sets";
	public static final String TABLE_NAME_CLASSIFIER_SET_MEMBERS = "classifier_set_members";
	public static final String TABLE_NAME_CLASSIFIER_RUNS = "classifier_runs";

	public static final String COLUMN_LABEL_CLASSIFIER_RUN_ID = "classifier_run_id";
	public static final String COLUMN_LABEL_CLASSIFIER_NAME = "classifier_name";
	public static final String COLUMN_LABEL_CLASSIFIER_CONFIGURATION = "classifier_configuration";
	public static final String COLUMN_LABEL_CLASSIFIER_PERFORMANCE = "predictive_accuracy";
	public static final String COLUMN_LABEL_CLASSIFIER_SET_NAME = "classifier_set_name";
	public static final String COLUMN_LABEL_CLASSIFIER_EVALUATION_METHOD = "classifier_evaluation_method";

	public static final String COLUMN_LABEL_JOB_STATUS = "status";
	public static final String JOB_STATUS_CREATED = "created";
	public static final String JOB_STATUS_RUNNING = "running";
	public static final String JOB_STATUS_FINISHED = "finished";
	public static final String JOB_STATUS_ERROR = "error";

	/**
	 * The string used to separate a classifier from its configuration
	 */
	public static final String CLASSIFIER_NAME_CONFIG_SEPARATOR = " with configuration: ";

	private String host;
	private String user;
	private String password;
	private String database;
	private CustomMySQLAdapter adapter;

	/**
	 * Creates a new MetaDataBaseConnection object that subsequently can be used to
	 * write data to and retrieve from the given data base. The given login data is
	 * used to create connections to the data base.
	 * 
	 * @param host
	 *            The host used for connecting to the data base
	 * @param user
	 *            The user used for connection to the data base
	 * @param password
	 *            The password used for connecting to the data base
	 * @param database
	 *            The data base this object is connected to
	 */
	public MetaDataDataBaseConnection(final String host, final String user, final String password,
			final String database) {
		this.host = host;
		this.user = user;
		this.password = password;
		this.database = database;
	}

	/**
	 * Gets the performance values of the classifiers contained in the given
	 * classifier set for the given data set. They are returned as an instances
	 * object, which contains one instance with the features representing
	 * classifiers. The first feature is the data set id.
	 * 
	 * @param datasetId
	 *            The data set identified by its id
	 * @param classifierSetName
	 *            The name of the classifier set
	 * @return The performance values of the classifiers on the data set as an
	 *         Instances object.
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public Instances getClassifierPerformancesForDataSet(Integer datasetId, String classifierSetName)
			throws SQLException {
		// Get available meta features
		List<String> classifierSetMembers = getMembersOfClassifierSet(classifierSetName);

		// Get meta feature data for data set
		String query = String.format(
				"SELECT * FROM (SELECT %s, %s, %s, %s FROM %s WHERE %s=?) AS selected_runs INNER JOIN (SELECT %s, %s FROM %s WHERE %s=?) AS selected_classifiers ON selected_runs.%s=selected_classifiers.%s  AND selected_runs.%s=selected_classifiers.%s",
				COLUMN_LABEL_DATASET_ID, COLUMN_LABEL_CLASSIFIER_NAME, COLUMN_LABEL_CLASSIFIER_CONFIGURATION,
				COLUMN_LABEL_CLASSIFIER_PERFORMANCE, TABLE_NAME_CLASSIFIER_RUNS, COLUMN_LABEL_DATASET_ID,
				COLUMN_LABEL_CLASSIFIER_NAME, COLUMN_LABEL_CLASSIFIER_CONFIGURATION, TABLE_NAME_CLASSIFIER_SET_MEMBERS,
				COLUMN_LABEL_CLASSIFIER_SET_NAME, COLUMN_LABEL_CLASSIFIER_NAME, COLUMN_LABEL_CLASSIFIER_NAME,
				COLUMN_LABEL_CLASSIFIER_CONFIGURATION, COLUMN_LABEL_CLASSIFIER_CONFIGURATION);

		openConnection();

		ResultSet resultSet = adapter.getResultsOfQuery(query, Arrays.asList(datasetId.toString(), classifierSetName));
		Instances result = getInstancesFromResultSetForClassifiers(resultSet, Arrays.asList(datasetId),
				classifierSetMembers, datasetId + "_" + classifierSetName + "_performanceValues");

		closeConnection();
		return result;
	}

	/**
	 * Get classifier performances for all members of a given data set set. They are
	 * returned as an Instances object with one instance's features representing a
	 * data sets classifier performance values. The first attribute is the openML
	 * data set id of the data set represented by the instance.
	 * 
	 * @param dataSetSetName
	 *            For which data sets to get classifier performance values
	 * @param classifierSetName
	 *            Which classifier performance values to get
	 * @return An instances object containing the classifier performances for the
	 *         data sets
	 * @throws SQLException
	 */
	public Instances getClassifierPerformancesForDataSetSet(String dataSetSetName, String classifierSetName)
			throws SQLException {
		// Get available meta features and data set ids
		List<String> classifierSetMembers = getMembersOfClassifierSet(classifierSetName);
		List<Integer> datasetIds = getMembersOfDatasetSet(dataSetSetName);

		// Get meta feature data for data set
		String query = String.format(
				"SELECT * FROM (SELECT %s, %s, %s, %s FROM %s WHERE %s IN (SELECT %s FROM %s WHERE %s=?)) AS selected_runs INNER JOIN (SELECT %s, %s FROM %s WHERE %s=?) AS selected_classifiers ON selected_runs.%s=selected_classifiers.%s  AND selected_runs.%s=selected_classifiers.%s",
				COLUMN_LABEL_DATASET_ID, COLUMN_LABEL_CLASSIFIER_NAME, COLUMN_LABEL_CLASSIFIER_CONFIGURATION,
				COLUMN_LABEL_CLASSIFIER_PERFORMANCE, TABLE_NAME_CLASSIFIER_RUNS, COLUMN_LABEL_DATASET_ID,
				COLUMN_LABEL_DATASET_ID, TABLE_NAME_DATASET_SET_MEMBERS, COLUMN_LABEL_DATASET_SET_NAME,
				COLUMN_LABEL_CLASSIFIER_NAME, COLUMN_LABEL_CLASSIFIER_CONFIGURATION, TABLE_NAME_CLASSIFIER_SET_MEMBERS,
				COLUMN_LABEL_CLASSIFIER_SET_NAME, COLUMN_LABEL_CLASSIFIER_NAME, COLUMN_LABEL_CLASSIFIER_NAME,
				COLUMN_LABEL_CLASSIFIER_CONFIGURATION, COLUMN_LABEL_CLASSIFIER_CONFIGURATION);

		openConnection();

		ResultSet resultSet = adapter.getResultsOfQuery(query, Arrays.asList(dataSetSetName, classifierSetName));
		Instances result = getInstancesFromResultSetForClassifiers(resultSet, datasetIds, classifierSetMembers,
				dataSetSetName + "_" + classifierSetName + "_performanceValues");

		closeConnection();
		return result;
	}

	/**
	 * Gets the meta features contained in the specified meta feature set for the
	 * data set with the given id. They are returned as an Instances object with one
	 * instance's features representing a data sets meta features. The first feature
	 * is the data set id.
	 * 
	 * @param datasetId
	 *            The data set id
	 * @param metaDataSetName
	 *            The name of the set of meta features
	 * @return The meta features for the data set as an Instances object
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public Instances getMetaDataSetForDataSet(Integer datasetId, String metaDataSetName) throws SQLException {
		// Get available classifiers
		List<String> metaDataSetMembers = getMembersOfMetadataSet(metaDataSetName);

		// Construct query
		String query = String.format(
				"SELECT %s, %s, %s FROM (SELECT %s, %s FROM %s WHERE %s=?) AS selected_runs INNER JOIN %s ON selected_runs.%s=%s.%s WHERE %s IN (SELECT %s FROM %s WHERE %s=?)",
				COLUMN_LABEL_DATASET_ID, COLUMN_LABEL_METAFEATURE_NAME, COLUMN_LABEL_METAFEATURE_VALUE,
				COLUMN_LABEL_METAFEATURE_RUN_ID, COLUMN_LABEL_DATASET_ID, TABLE_NAME_METAFEATURE_RUNS,
				COLUMN_LABEL_DATASET_ID, TABLE_NAME_METAFEATURE_VALUES, COLUMN_LABEL_METAFEATURE_RUN_ID,
				TABLE_NAME_METAFEATURE_VALUES, COLUMN_LABEL_METAFEATURE_RUN_ID, COLUMN_LABEL_METAFEATURE_NAME,
				COLUMN_LABEL_METAFEATURE_NAME, TABLE_NAME_METAFEATURE_SET_MEMBERS, COLUMN_LABEL_METAFEATURE_SET_NAME);

		openConnection();

		ResultSet resultSet = adapter.getResultsOfQuery(query, Arrays.asList(datasetId.toString(), metaDataSetName));
		Instances result = getInstancesFromResultSetForMetaData(resultSet, Arrays.asList(datasetId), metaDataSetMembers,
				datasetId + "_" + metaDataSetName + "_metafeatures");

		closeConnection();
		return result;
	}

	/**
	 * Get a set of meta data for all members of a given data set set. The first
	 * attribute of the returned instances object is the openML data set id of the
	 * data set that instance contains the meta feature values for.
	 * 
	 * @param dataSetSetName
	 *            The data sets for which to get the meta features
	 * @param metaDataSetSetName
	 *            Which meta features to get for the data set
	 * @return A new instances object containing instances that represent meta
	 *         features for a data set
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public Instances getMetaDataSetForDataSetSet(String dataSetSetName, String metaDataSetSetName) throws SQLException {
		List<Integer> datasetIds = getMembersOfDatasetSet(dataSetSetName);
		List<String> metaDataSetMembers = getMembersOfMetadataSet(metaDataSetSetName);

		String query = String.format(
				"SELECT %s, %s, %s FROM (SELECT %s, %s FROM %s WHERE %s IN (SELECT %s FROM %s WHERE %s=?)) AS selected_runs INNER JOIN %s ON selected_runs.%s=%s.%s WHERE %s IN (SELECT %s FROM %s WHERE %s=?) ORDER BY %s",
				COLUMN_LABEL_DATASET_ID, MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_VALUE,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_RUN_ID, COLUMN_LABEL_DATASET_ID,
				MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_RUNS, COLUMN_LABEL_DATASET_ID,
				COLUMN_LABEL_DATASET_ID, TABLE_NAME_DATASET_SET_MEMBERS, COLUMN_LABEL_DATASET_SET_NAME,
				MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_VALUES,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_RUN_ID,
				MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_VALUES,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_RUN_ID,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME,
				MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_SET_MEMBERS,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_SET_NAME, COLUMN_LABEL_DATASET_ID);

		openConnection();

		ResultSet resultSet = adapter.getResultsOfQuery(query, Arrays.asList(dataSetSetName, metaDataSetSetName));
		Instances result = getInstancesFromResultSetForMetaData(resultSet, datasetIds, metaDataSetMembers,
				dataSetSetName + "_" + metaDataSetSetName);

		closeConnection();

		return result;
	}

	/**
	 * Gets the classifier names together with their configuration that are
	 * contained in the given set. They are separated by
	 * {@link MetaDataDataBaseConfiguration#CLASSIFIER_NAME_CONFIG_SEPARATOR}. They
	 * are ordered by classifier name first and then configuration (alphabetically).
	 * 
	 * @param setName
	 *            The name of the set of classifiers
	 * @return The members of the set of classifiers
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public ArrayList<String> getMembersOfClassifierSet(String setName) throws SQLException {
		openConnection();
		// Formulate query
		String query = String.format("SELECT %s,%s FROM %s WHERE %s=? ORDER BY %s, %s", COLUMN_LABEL_CLASSIFIER_NAME,
				COLUMN_LABEL_CLASSIFIER_CONFIGURATION, TABLE_NAME_CLASSIFIER_SET_MEMBERS,
				COLUMN_LABEL_CLASSIFIER_SET_NAME, COLUMN_LABEL_CLASSIFIER_NAME, COLUMN_LABEL_CLASSIFIER_CONFIGURATION);
		ResultSet resultSet = adapter.getResultsOfQuery(query, Arrays.asList(setName));

		// Collect members
		ArrayList<String> classifierSetMembers = new ArrayList<String>();
		while (resultSet.next()) {
			String classifierName = resultSet.getString(COLUMN_LABEL_CLASSIFIER_NAME);
			String classifierConfig = resultSet.getString(COLUMN_LABEL_CLASSIFIER_CONFIGURATION);
			classifierSetMembers.add(classifierName + CLASSIFIER_NAME_CONFIG_SEPARATOR + classifierConfig);
		}
		closeConnection();

		return classifierSetMembers;
	}

	/**
	 * Gets the names of all the meta features contained in this set of meta
	 * features. They are ordered alphabetically.
	 * 
	 * @param setName
	 *            The name of the meta feature set
	 * @return The names of the members of the meta feature set
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public ArrayList<String> getMembersOfMetadataSet(String setName) throws SQLException {
		openConnection();
		// Formulate query
		String query = String.format("SELECT %s FROM %s WHERE %s=? ORDER BY %s",
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME,
				MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_SET_MEMBERS,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_SET_NAME,
				MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME);
		ResultSet resultSet = adapter.getResultsOfQuery(query, Arrays.asList(setName));

		// Collect members
		ArrayList<String> metaDataSetMembers = new ArrayList<String>();
		while (resultSet.next()) {
			String metafeatureName = resultSet.getString(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME);
			metaDataSetMembers.add(metafeatureName);
		}

		closeConnection();
		return metaDataSetMembers;
	}

	/**
	 * Gets the ids of the data sets contained in the given data set set ordered by
	 * id.
	 * 
	 * @param setName
	 *            The name of the data set set
	 * @return The ids of of the data sets contained in the data set set
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public ArrayList<Integer> getMembersOfDatasetSet(String setName) throws SQLException {
		openConnection();

		// Formulate query
		String query = String.format("SELECT %s FROM %s WHERE %s=? ORDER BY %s", COLUMN_LABEL_DATASET_ID,
				TABLE_NAME_DATASET_SET_MEMBERS, COLUMN_LABEL_DATASET_SET_NAME, COLUMN_LABEL_DATASET_ID);
		ResultSet resultSet = adapter.getResultsOfQuery(query, Arrays.asList(setName));

		// Collect members
		ArrayList<Integer> metaDataSetMembers = new ArrayList<Integer>();
		while (resultSet.next()) {
			int datasetId = resultSet.getInt(COLUMN_LABEL_DATASET_ID);
			metaDataSetMembers.add(datasetId);
		}

		closeConnection();
		return metaDataSetMembers;
	}

	/**
	 * Gets a list of all available sets of classifiers.
	 * 
	 * @return The list of available classifier sets
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public List<String> getAvailableClassifierSets() throws SQLException {
		return getAvailableSets(COLUMN_LABEL_CLASSIFIER_SET_NAME, TABLE_NAME_CLASSIFIER_SETS);
	}

	/**
	 * Gets a list of all available sets of meta features.
	 * 
	 * @return The list of available meta features sets
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public List<String> getAvailableMetaDataSets() throws SQLException {
		return getAvailableSets(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_SET_NAME,
				MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_SETS);
	}

	/**
	 * Gets a list of all available sets of data sets.
	 * 
	 * @return The list of available data set sets
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public List<String> getAvailableDataSetSets() throws SQLException {
		return getAvailableSets(COLUMN_LABEL_DATASET_SET_NAME, TABLE_NAME_DATASET_SETS);
	}

	/**
	 * Adds computed performance data of the classifier to the data base. A new run
	 * is automatically created for the data. The classifier and its configuration
	 * are assumed to be condensed to a String, separated by
	 * {@link MetaDataDataBaseConfiguration#CLASSIFIER_NAME_CONFIG_SEPARATOR}.
	 * 
	 * @param datasetId
	 *            The data set for which the classifier performance has been
	 *            computed
	 * @param classifierWithConfig
	 *            The classifier name with configuration
	 * @param evaluationMethod
	 *            The method by which the classifier has been evaluated
	 * @param performance
	 *            The performance of the classifier
	 * @throws SQLException
	 *             if something goes wrong while connecting to the data base
	 */
	public void addClassifierPerformance(int datasetId, String classifierWithConfig, String evaluationMethod,
			double performance) throws SQLException {
		int runId = addClassifierExperiment(datasetId, classifierWithConfig, evaluationMethod);
		addClassifierPerformanceForRun(runId, performance);
	}

	/**
	 * Adds computed performance data of the classifier to the data base. A run with
	 * all the information has to have been created for this before.
	 * 
	 * @param runId
	 *            The run id of the classifier run
	 * @param performance
	 *            The performance of the classifier for this run
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data bse
	 */
	public void addClassifierPerformanceForRun(int runId, double performance) throws SQLException {
		openConnection();

		// insert performance, update job status
		HashMap<String, Object> conditions = new HashMap<String, Object>();
		conditions.put(COLUMN_LABEL_CLASSIFIER_RUN_ID, runId);
		HashMap<String, Object> updateValues = new HashMap<String, Object>();
		updateValues.put(COLUMN_LABEL_CLASSIFIER_PERFORMANCE, Double.isNaN(performance) ? -1 : performance);
		updateValues.put(COLUMN_LABEL_JOB_STATUS, JOB_STATUS_FINISHED);
		try {
			adapter.update(TABLE_NAME_CLASSIFIER_RUNS, updateValues, conditions);
		} catch (SQLException e) {
			System.err.println(
					"Could not add classifier performance for run " + runId + " with performance value " + performance);
			throw e;
		}

		closeConnection();
	}

	/**
	 * Creates a new classifier experiment in the data base with the given
	 * parameters. The classifier name and configuration are assumed to be separated
	 * by {@link MetaDataDataBaseConfiguration#CLASSIFIER_NAME_CONFIG_SEPARATOR}.
	 * The classifier configuration is assumed to have been serialized by
	 * {@link mySQLHelper#convertOptionsStringToArray(String)}.
	 * 
	 * @param datasetId
	 *            The data set on which the classifier should be run
	 * @param classifierWithConfig
	 *            The classifier name and configuration
	 * @param evaluationMethod
	 *            The method by which the classifier performance should be evaluated
	 * @return The run id of the newly created run
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public int addClassifierExperiment(int datasetId, String classifierWithConfig, String evaluationMethod)
			throws SQLException {
		openConnection();

		// get classifier name and configuration
		String[] classifierNameAndConfig = classifierWithConfig.split(CLASSIFIER_NAME_CONFIG_SEPARATOR);

		// add new run
		HashMap<String, Object> map = new HashMap<String, Object>();
		map.put(COLUMN_LABEL_DATASET_ID, datasetId);
		map.put(COLUMN_LABEL_CLASSIFIER_NAME, classifierNameAndConfig[0]);
		map.put(COLUMN_LABEL_CLASSIFIER_CONFIGURATION, classifierNameAndConfig[1]);
		map.put(COLUMN_LABEL_CLASSIFIER_EVALUATION_METHOD, evaluationMethod);
		int runId;
		try {
			runId = adapter.insert(TABLE_NAME_CLASSIFIER_RUNS, map);
		} catch (SQLException e) {
			System.err.println("Could not add classifier experiment: datasetId " + datasetId + " classifier "
					+ classifierWithConfig + " evaluation method " + evaluationMethod);
			throw e;
		}

		closeConnection();

		return runId;
	}

	/**
	 * Adds computed meta data to the data base. If the runId is left as null, a new
	 * run is automatically created for the data.
	 * 
	 * @param datasetId
	 *            The id of the data set for which the meta data has been computed
	 * @param featureValues
	 *            The values for the meta feature that have been computed
	 * @param groupTimes
	 *            The times for the computation of the meta feature groups
	 * @param runId
	 *            The run id of the run (if exists)
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public void addMetaDataForDataSet(int datasetId, HashMap<String, Double> featureValues,
			HashMap<String, Double> groupTimes, Integer runId) throws SQLException {

		int createdRunId;
		if (runId == null) {
			createdRunId = addMetaFeatureExperiment(datasetId);
		} else {
			createdRunId = runId;
		}

		openConnection();

		// insert times
		HashMap<String, Object> map = new HashMap<String, Object>();
		map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_RUN_ID, createdRunId);
		for (Map.Entry<String, Double> entry : groupTimes.entrySet()) {
			map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_GROUP_NAME, entry.getKey());
			map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_COMPUTATION_TIME, entry.getValue());
			adapter.insert_noNewValues(MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_TIMES, map);
		}

		// insert values
		map = new HashMap<String, Object>();
		map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_RUN_ID, createdRunId);
		for (Map.Entry<String, Double> entry : featureValues.entrySet()) {
			map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME, entry.getKey());
			double metafeatureValue = entry.getValue();
			map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_VALUE,
					Double.isNaN(metafeatureValue) ? -1 : metafeatureValue);
			adapter.insert_noNewValues(MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_VALUES, map);
		}

		// update job status
		HashMap<String, Object> conditions = new HashMap<String, Object>();
		conditions.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_RUN_ID, createdRunId);
		HashMap<String, String> updateValues = new HashMap<String, String>();
		updateValues.put(COLUMN_LABEL_JOB_STATUS, JOB_STATUS_FINISHED);
		adapter.update(MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_RUNS, updateValues, conditions);

		closeConnection();
	}

	/**
	 * Creates a new run in the table of meta feature runs (no other info attached).
	 * 
	 * @param datasetId
	 *            The id of the data set for which meta feature are to be computed
	 * @return The id of the just created run
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public int addMetaFeatureExperiment(int datasetId) throws SQLException {
		openConnection();

		// add new run
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		map.put(COLUMN_LABEL_DATASET_ID, datasetId);
		int runId = adapter.insert(MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_RUNS, map);

		closeConnection();

		return runId;
	}

	/**
	 * Adds the given set identified by its name to the list of classifier sets and
	 * all the members as its members.
	 * 
	 * @param setName
	 *            The name of the new classifier set
	 * @param members
	 *            The members of the new classifier set
	 * @throws SQLException
	 *             If something goes wrong with the data base connection
	 */
	public void addClassifierSet(String setName, List<AbstractClassifier> members) throws SQLException {
		openConnection();

		// add set name to sets
		HashMap<String, String> map = new HashMap<String, String>();
		map.put(COLUMN_LABEL_CLASSIFIER_SET_NAME, setName);
		adapter.insert_noNewValues(TABLE_NAME_CLASSIFIER_SETS, map);

		// add members to set
		map = new HashMap<String, String>();
		map.put(COLUMN_LABEL_CLASSIFIER_SET_NAME, setName);
		for (AbstractClassifier member : members) {
			map.put(COLUMN_LABEL_CLASSIFIER_NAME, member.getClass().getName());
			map.put(COLUMN_LABEL_CLASSIFIER_CONFIGURATION, Arrays.toString(member.getOptions()));
			adapter.insert_noNewValues(TABLE_NAME_CLASSIFIER_SET_MEMBERS, map);
		}
		closeConnection();
	}

	/**
	 * Adds the given set identified by its name to the list of meta data sets and
	 * all the members as its members.
	 * 
	 * @param setName
	 *            The name of the new meta data set
	 * @param members
	 *            The members of the new meta data set
	 * @throws SQLException
	 *             If something goes wrong with the data base connection
	 */
	public void addMetaDataSet(String setName, List<String> members) throws SQLException {
		openConnection();

		// add set name to sets
		HashMap<String, String> map = new HashMap<String, String>();
		map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_SET_NAME, setName);
		adapter.insert_noNewValues(MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_SETS, map);

		// add members to set
		map = new HashMap<String, String>();
		map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_SET_NAME, setName);
		for (String member : members) {
			map.put(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME, member);
			adapter.insert_noNewValues(MetaDataDataBaseConnection.TABLE_NAME_METAFEATURE_SET_MEMBERS, map);
		}

		closeConnection();
	}

	/**
	 * Add a set of data sets to the data base.
	 * 
	 * @param setName
	 *            The name of the new data set set
	 * @param members
	 *            The members of the new data set set (data set ids)
	 * @throws SQLException
	 *             If something goes wrong while connecting to the data base
	 */
	public void addDatasetSet(String setName, List<Integer> members) throws SQLException {
		openConnection();

		// add set name to sets
		HashMap<String, String> map = new HashMap<String, String>();
		map.put(COLUMN_LABEL_DATASET_SET_NAME, setName);
		adapter.insert_noNewValues(TABLE_NAME_DATASET_SETS, map);

		// add members to set
		map = new HashMap<String, String>();
		map.put(COLUMN_LABEL_DATASET_SET_NAME, setName);
		for (Integer member : members) {
			map.put(COLUMN_LABEL_DATASET_ID, member.toString());
			adapter.insert_noNewValues(TABLE_NAME_DATASET_SET_MEMBERS, map);
		}

		closeConnection();
	}
	
	/**
	 * Add groups of meta features to the data base.
	 * 
	 * @param metafeatureGroups The meta feature group names together with the features
	 * @param host The host for the MySQL connection
	 * @param user The user name for the MySQL connection
	 * @param pw The password for the MySQL connection
	 * @param database The data base for the MySQL connection
	 */
	public void addMetaFeatureGroup(Map<String,List<String>> metafeatureGroups) {
		openConnection();
		
		Map<String,String> map = new HashMap<String,String>();
		metafeatureGroups.forEach((name, list) -> {
			map.clear();
			list.forEach(elem -> {
				map.put("metafeature_group", name);
				map.put("metafeature_name", elem);
				try {
					adapter.insert_noNewValues("metafeature_groups", map);
				} catch (SQLException e) {
					throw new RuntimeException(e.getMessage());
				}
				System.out.println(name + " " + elem);
			});

		});
		
		closeConnection();
	}
	
	public static String[] convertOptionsStringToArray(String serializedOptions) {
		return Arrays.stream(serializedOptions.substring(1, serializedOptions.length()-1).split(",")).map(String::trim).toArray(String[]::new);
	}
	
	public static String convertOptionsArrayToString(String[] options) {
		return Arrays.toString(options);
	}

	private Instances getInstancesFromResultSetForClassifiers(ResultSet resultSet, List<Integer> datasetIds,
			List<String> classifierSetMembers, String instancesName) throws SQLException {
		// Create instances
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute(COLUMN_LABEL_DATASET_ID));
		classifierSetMembers.forEach(member -> attributes.add(new Attribute(member)));
		Instances instances = new Instances(instancesName, attributes, 0);

		// Create instance for each data set with id
		TreeMap<Integer, DenseInstance> instancesForDataSets = new TreeMap<Integer, DenseInstance>();
		datasetIds.forEach(datasetId -> {
			double[] values = new double[attributes.size()];
			Arrays.fill(values, Double.NaN);
			values[0] = datasetId;
			DenseInstance instance = new DenseInstance(1, values);
			instancesForDataSets.put(datasetId, instance);
		});

		// Gather results
		while (resultSet.next()) {
			Integer dataSetId = resultSet.getInt(COLUMN_LABEL_DATASET_ID);
			String classifierName = resultSet.getString(COLUMN_LABEL_CLASSIFIER_NAME);
			String classifierConfiguration = resultSet.getString(COLUMN_LABEL_CLASSIFIER_CONFIGURATION);
			Double performanceValue = resultSet.getDouble(COLUMN_LABEL_CLASSIFIER_PERFORMANCE);
			performanceValue = performanceValue == -1 ? Double.NaN : performanceValue;
			instancesForDataSets.get(dataSetId).setValue(instances
					.attribute(classifierName + CLASSIFIER_NAME_CONFIG_SEPARATOR + classifierConfiguration).index(),
					performanceValue);
		}

		for (Integer datasetId : instancesForDataSets.navigableKeySet()) {
			instances.add(instancesForDataSets.get(datasetId));
		}

		return instances;
	}

	private Instances getInstancesFromResultSetForMetaData(ResultSet resultSet, List<Integer> datasetIds,
			List<String> metaDataSetMembers, String instancesName) throws SQLException {
		// Create instances
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute(COLUMN_LABEL_DATASET_ID));
		metaDataSetMembers.forEach(member -> attributes.add(new Attribute(member)));
		Instances instances = new Instances(instancesName, attributes, 0);

		// Create instance for each data set with id
		TreeMap<Integer, DenseInstance> instancesForDataSets = new TreeMap<Integer, DenseInstance>();
		datasetIds.forEach(datasetId -> {
			double[] values = new double[attributes.size()];
			Arrays.fill(values, Double.NaN);
			values[0] = datasetId;
			DenseInstance instance = new DenseInstance(1, values);
			instancesForDataSets.put(datasetId, instance);
		});

		// Gather results
		while (resultSet.next()) {
			Integer dataSetId = resultSet.getInt(COLUMN_LABEL_DATASET_ID);
			String metafeatureName = resultSet.getString(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_NAME);
			Double metafeatureValue = resultSet.getDouble(MetaDataDataBaseConnection.COLUMN_LABEL_METAFEATURE_VALUE);
			metafeatureValue = metafeatureValue == -1 ? Double.NaN : metafeatureValue;
			instancesForDataSets.get(dataSetId).setValue(instances.attribute(metafeatureName).index(),
					metafeatureValue);
		}

		for (Integer datasetId : instancesForDataSets.navigableKeySet()) {
			instances.add(instancesForDataSets.get(datasetId));
		}

		return instances;
	}

	private List<String> getAvailableSets(String setNameColumn, String table) throws SQLException {
		// Formulate query
		String query = String.format("SELECT DISTINCT %s FROM %s", setNameColumn, table);

		openConnection();

		ResultSet resultSet = adapter.getResultsOfQuery(query);

		// Enumerate results
		List<String> availableSets = new ArrayList<String>();
		while (resultSet.next()) {
			String setName = resultSet.getString(setNameColumn);
			availableSets.add(setName);
		}

		closeConnection();

		return availableSets;
	}

	private void openConnection() {
		adapter = new CustomMySQLAdapter(host, user, password, database);
	}

	private void closeConnection() {
		adapter.close();
	}
}
