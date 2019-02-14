package com.nju.imagerec.cnn;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.Arrays;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class Reduce extends Reducer<Text, DoubleWritable, Text, DoubleWritable>{
	double [] b1=new double[32];
	double [] b2=new double[10];
	double [][] w1=new double[32][784];
	double [][] w2=new double[10][32];
	
	public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
		Integer totalOccurrances = new Integer(0);
		Double totalValue = new Double(0.0);
		for (DoubleWritable value : values) {
			String v = value.toString();
			totalOccurrances = totalOccurrances + 1; 
			totalValue = totalValue + Double.parseDouble(v);
		}
		Double tot = new Double(totalOccurrances);
		Double average = totalValue/tot;
		
		String [] keySplit = key.toString().split("-");
		
		if(keySplit[0].equals("b1")) {
			b1[Integer.parseInt(keySplit[1])] = average;
		}
		
		else if(keySplit[0].equals("b2")) {
			b2[Integer.parseInt(keySplit[1])] = average;
		}
		
		else if(keySplit[0].equals("w1")) {
			w1[Integer.parseInt(keySplit[1])][Integer.parseInt(keySplit[2])] = average;
		}
		
		else if(keySplit[0].equals("w2")) {
			w2[Integer.parseInt(keySplit[1])][Integer.parseInt(keySplit[2])] = average;
		}
	}
	public void cleanup(Context context) throws IOException, InterruptedException {
		URI[] cacheFiles = context.getCacheFiles();
		  if (cacheFiles != null && cacheFiles.length > 0) {
		    try {
			    int n_x=784;
			    int n_h=32;
			    int n_y=10;
			    NeuralNetwork neuralNetwork = new NeuralNetwork();
			    neuralNetwork.initialize_with_zeros(n_x,n_h,n_y);
		    		neuralNetwork.w1 = w1;
		    		neuralNetwork.w2 = w2;
		    		neuralNetwork.b1 = b1;
		    		neuralNetwork.b2 = b2;
		    		
		        FileSystem fs = FileSystem.get(context.getConfiguration());
		        Path path = new Path(cacheFiles[0].toString());
		        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));		        
		        String s;
		        Integer correct = 0;
		        Integer total = 0;
		        while ((s=reader.readLine()) != null) {
		        	
			        	String sample = s.toString();
			    		StringTokenizer stringTokenizer = new StringTokenizer(sample);
			    		
			    		Double label = 0.0;
	
			    		int i = 0;
			    		double [] imageTest = new double [784];
			    		while (stringTokenizer.hasMoreTokens()) {
			    			if (i == 784) {
			    				label = Double.parseDouble(stringTokenizer.nextToken());
			    			}
			    			else {
			    				String term = stringTokenizer.nextToken();
			    				imageTest[i] = Double.parseDouble(term);
			    			}
			    			i++;
			    		}
			    		double [] vectorImage = neuralNetwork.narmalize_data(imageTest);
		    			neuralNetwork.forward_propagation(vectorImage);
		    			double predict_value = neuralNetwork.softmax(neuralNetwork.a2);
		    			Integer actualLabel = label.intValue();
		    			if((int)predict_value == actualLabel){
		    				correct++;
		    			}
		    			total++;
		         }
		        System.out.println("Correct: " + correct.toString());
		        Double correctDoub = Double.valueOf(correct);
		        Double totalDoub = Double.valueOf(total);
		        System.out.println("Accuracy: " + (new Double (correctDoub/totalDoub)).toString());
		    } 
		    finally {
		    		System.out.println("Complete");
		    }
		}
	}
}
