package com.nju.imagerec.cnn;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class Map extends Mapper<LongWritable, Text, Text, DoubleWritable>{
	
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException, NoSuchElementException {
		
		String sample = value.toString();
		StringTokenizer stringTokenizer = new StringTokenizer(sample);
		StringBuilder X = new StringBuilder(); 
		
		Double label = 0.0;
//		double [] imageTrain = new double [784];
		
		int i = 0;
		while (stringTokenizer.hasMoreTokens()) {
			if (i == 784) {
				label = Double.parseDouble(stringTokenizer.nextToken());
			}
			else {
				X.append(stringTokenizer.nextToken());
//				imageTrain[i] = Double.parseDouble(stringTokenizer.nextToken());
			}
			i++;
		}
	
		Text newkeyText = new Text (X.toString());
		DoubleWritable newVal = new DoubleWritable (label);
		
		context.write(newkeyText, newVal);
	}
	
}
