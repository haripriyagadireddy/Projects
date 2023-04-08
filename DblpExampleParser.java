import java.io.*;
import java.sql.*;
import java.util.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.mysql.cj.PreparedQuery;
import org.dblp.mmdb.*;
import org.xml.sax.SAXException;
@SuppressWarnings("javadoc")
class DlpExampleParser {

    public static void main(String[] args) throws IOException {
                // we need to raise entityExpansionLimit because the dblp.xml has millions of entities
        System.setProperty("entityExpansionLimit", "10000000");

        if (args.length != 2) {
            System.err.format("Usage: java %s <dblp-xml-file> <dblp-dtd-file>\n", DlpExampleParser.class.getName());
            System.exit(0);
        }
        //arguments to pass xml and dtd file
        String dblpXmlFilename = args[0];
        String dblpDtdFilename = args[1];
        System.out.println("building the dblp main memory DB ...");
        RecordDbInterface dblp;
        try {
            dblp = new RecordDb(dblpXmlFilename, dblpDtdFilename, false);
        } catch (final IOException ex) {
            System.err.println("cannot read dblp XML: " + ex.getMessage());
            return;
        } catch (final SAXException ex) {
            System.err.println("cannot parse XML: " + ex.getMessage());
            return;
        }
        //Connection to mysql
        DblpStatements(dblp);
       // AuthorTable(dblp);
       // Author_key(dblpXmlFilename);
        System.out.format("MMDB ready: %d publs, %d pers\n\n", dblp.numberOfPublications(), dblp.numberOfPersons());
        System.out.println("done.");
    }
    private static Map<String, List<String>> searchDirectories(String keyToBeSearched) throws IOException, FileNotFoundException {
        String fileName = keyToBeSearched.substring(keyToBeSearched.lastIndexOf("/") + 1, keyToBeSearched.length());
        String pathToSearch = "/mnt/hdd/gadireddy/merged-fixed-aux/" + keyToBeSearched.substring(0, keyToBeSearched.indexOf(fileName));
        FileInputStream inputStream = null;
        try {
            try {
                 inputStream = new FileInputStream(pathToSearch + fileName);
            }
            catch(FileNotFoundException ex){
                inputStream = new FileInputStream(pathToSearch + fileName + ".xml");
            }
            StringBuilder textBuilder = new StringBuilder();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = reader.readLine()) != null) {
                textBuilder.append(line);
                textBuilder.append("\n");
            }
            Map<String, List<String>> responseData = extractContent(textBuilder);
            reader.close();
            return responseData;
        } catch (FileNotFoundException ex) {
           System.out.println(pathToSearch + fileName + " This File is not available");
            }
        return null;
    }


    private static Map<String, List<String>> extractContent(StringBuilder textBuilder) {
        Map<String, List<String>> responseMap = new HashMap<>();
        List<String> responseList = new ArrayList<>();
        List<String> citedByList = new ArrayList<>();
        List<String> citeList = new ArrayList<>();
        if (null != textBuilder) {
            String[] response = textBuilder.toString().split("\n");
            for (int i = 0; i <= response.length - 1; i++) {
                if (response[i].contains("<abstract")) {
                    if (response[i + 1].contains("<![CDATA[")) {
                        if (response[i+1].contains("<citerefgrp")) {
                            responseList.add(response[i + 1].substring(response[i + 1].indexOf("<![CDATA[") + 9,
                                    response[i + 1].indexOf("<citerefgrp"))
                                    + response[i + 1].substring(response[i + 1].indexOf("</citerefgrp") + 13,
                                    response[i + 1].indexOf("]]>")));
                        }else {
                            responseList.add(response[i + 1].substring(response[i + 1].indexOf("<![CDATA[") + 9, response[i + 1].indexOf("]]>")));
                        }
                    }
                }
                if (response[i].contains("<cite")) {
                    if (response[i].contains("<citedBy")) {
                        citedByList.add(response[i].substring(response[i].indexOf("key=") + 5, response[i].indexOf("\" src_id=\"")));
                    } else {
                        citeList.add(response[i].substring(response[i].indexOf("key=") + 5, response[i].indexOf("\" src_id=\"")));
                    }
                }
            }
            responseMap.put("CITEDBY", citedByList);
            responseMap.put("CITE", citeList);
            responseMap.putIfAbsent("Abstract", responseList);
        }
        return responseMap;
    }
public static void AuthorTable(RecordDbInterface dblp){
        try{
            String Author_table = "CREATE TABLE IF NOT EXISTS `Author_table`"+
                    "(`Id` int not null auto_increment,`keytobesearched` VARCHAR(500) binary not null,"+
                    " `Author` longtext, primary key(Id)) ENGINE = InnoDB DEFAULT CHARSET=utf8mb4;";
            String INSERT_Author_SQL = "INSERT INTO Author_table (`keytobesearched` ,`Author` )VALUES  (?, ?);";

            Class.forName("com.mysql.cj.jdbc.Driver");
            Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/gadireddy?autoReconnect=true&useSSL=false", "gadireddy", "Ipzkb1jCXJSSCM37sOSJ");
            System.out.print("connected");
            Statement smt=connection.createStatement();
            smt.execute(Author_table);
            System.out.println("created table");
            PreparedStatement preparedStatement = connection.prepareStatement(INSERT_Author_SQL);
            connection.setAutoCommit(true);
            Collection<Publication> pubs = dblp.getPublications();
            Iterator<Publication> pubIt = pubs.iterator();

            int i = 0;
            String keyToBeSearched;
            while (pubIt.hasNext()) {

                Publication currPub = pubIt.next();
                Iterator<Field> pubFields = currPub.getFields().iterator();
                keyToBeSearched = currPub.getKey();
                preparedStatement.setString(1, keyToBeSearched);
                System.out.println(keyToBeSearched + "\n");
                // for title we search all fields
                while (pubFields.hasNext()) {
                    Field currFieldInPub = pubFields.next();

                    if (currFieldInPub.tag().contains("author") ) {
                        preparedStatement.setString(2, currFieldInPub.value());
                        preparedStatement.execute();
                    }
                }
                System.out.println(" ---" + i + "\n");
                i++;

            }
            connection.close();
        } catch (ClassNotFoundException | SQLException ex) {
            ex.printStackTrace();
        }
}
public static void Author_key(String dblpXmlFilename){
    try{

        String Author_key = "CREATE TABLE IF NOT EXISTS `Author_key`"+
                "(`key` VARCHAR(500) binary not null,"+
                " `author` longtext, primary key(`key`)) ENGINE = InnoDB DEFAULT CHARSET=utf8mb4;";
        String local_file="set global local_infile=true;";
        String INSERT_Authorkey_SQL = "LOAD XML LOCAL INFILE '"+dblpXmlFilename+"' INTO TABLE Author_key ROWS IDENTIFIED BY '<www>';";


        Class.forName("com.mysql.cj.jdbc.Driver");
        Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/gadireddy?autoReconnect=true&useSSL=false", "gadireddy", "Ipzkb1jCXJSSCM37sOSJ");
        System.out.print("connected");
        Statement smt=connection.createStatement();
        smt.execute(Author_key);
        System.out.println("created table");
        //smt.execute(local_file);
        smt.execute(INSERT_Authorkey_SQL);
        connection.setAutoCommit(true);

        connection.close();
    } catch (ClassNotFoundException | SQLException ex) {
        ex.printStackTrace();
    }

}



public static void DblpStatements(RecordDbInterface dblp){
    try {
        //Statement to create the data table
        String Dblp ="CREATE TABLE IF NOT EXISTS `Dblp`"+
                "(`keytobesearched` VARCHAR(500) binary not null,`title` longtext ,  `abstract` longtext ,  `citeby` longtext,`cite` longtext,"+
                "primary key(keytobesearched)) ENGINE = InnoDB DEFAULT CHARSET=utf8mb4;";
        //Insert statement for the data table
        String INSERT_USERS_SQL = "INSERT INTO Dblp VALUES  (?, ?,?,?,?);";
        Class.forName("com.mysql.cj.jdbc.Driver");
        Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/gadireddy?autoReconnect=true&useSSL=false", "gadireddy", "Ipzkb1jCXJSSCM37sOSJ");
        System.out.print("connected");
        Statement smt=connection.createStatement();
        smt.execute(Dblp);
        System.out.println("created table");
        PreparedStatement preparedStatement = connection.prepareStatement(INSERT_USERS_SQL);
        connection.setAutoCommit(true);
        Collection<Publication> pubs = dblp.getPublications();
        Iterator<Publication> pubIt = pubs.iterator();
        int i = 0;
        String keyToBeSearched;
        while (pubIt.hasNext()) {
            Publication currPub = pubIt.next();
            Iterator<Field> pubFields = currPub.getFields().iterator();
            keyToBeSearched = currPub.getKey();
            preparedStatement.setString(1, keyToBeSearched);
            System.out.println(keyToBeSearched + "\n");
            // for title we search all fields
            while (pubFields.hasNext()) {
                int p=0;
                Field currFieldInPub = pubFields.next();
                if (currFieldInPub.tag().contains("title") && !currFieldInPub.tag().contains("booktitle")) {
                    preparedStatement.setString(2, currFieldInPub.value());
                    p=1;
                }
                else if (p==1){
                    break;
                }
            }
            try {
                Map<String, List<String>> responseData = searchDirectories(keyToBeSearched);
                if (responseData != null) {
                    int k = 3;
                    for (List<String> entry : responseData.values()) {
                        if (entry.size() == 0)
                        {
                            preparedStatement.setString(k, null);
                        }
                        else{
                            preparedStatement.setString(k, entry.toString());
                        }
                        k++;
                    }
                } else {
                    preparedStatement.setString(3, null);
                    preparedStatement.setString(4, null);
                    preparedStatement.setString(5, null);
                }
            } catch (IOException e) {
                System.out.println("null list");
            }
            System.out.println(" ---" + i + "\n");
            i++;
            preparedStatement.execute();
        }
        connection.close();
    } catch (ClassNotFoundException | SQLException ex) {
        ex.printStackTrace();
    }
    }
}