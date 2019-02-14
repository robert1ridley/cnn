package com.nju.imagerec.cnn;

import java.io.IOException;
import java.util.Arrays;

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
			System.out.println("b2");
		}
		
		else if(keySplit[0].equals("w1")) {
			w1[Integer.parseInt(keySplit[1])][Integer.parseInt(keySplit[2])] = average;
		}
		
		else if(keySplit[0].equals("w2")) {
			w2[Integer.parseInt(keySplit[1])][Integer.parseInt(keySplit[2])] = average;
		}

		context.write(key, new DoubleWritable(totalOccurrances));
	}
	public void cleanup(Context context) throws IOException, InterruptedException {
		System.out.println(Arrays.toString(w1[0]));
	}
}
