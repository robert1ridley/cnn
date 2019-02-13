package com.nju.imagerec.cnn;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class Reduce extends Reducer<Text, DoubleWritable, Text, DoubleWritable>{
	
	Double totalOccurrances = 0.0;
	public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
		for (DoubleWritable value : values) {
			totalOccurrances = totalOccurrances + Double.parseDouble("1.0");
		}
		DoubleWritable wordCount = new DoubleWritable(totalOccurrances);
//		context.write(key, wordCount);
	}
	public void cleanup(Context context) throws IOException, InterruptedException {
		DoubleWritable wordCount = new DoubleWritable(totalOccurrances);
		context.write(new Text("Result: "), wordCount);
	}
}
