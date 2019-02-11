package com.nju.imagerec.cnn;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class Reduce extends Reducer<Text, IntWritable, Text, IntWritable>{
	public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
		int totalOccurrances = 0;
		for (IntWritable value : values) {
			totalOccurrances = totalOccurrances + value.get();
		}
		IntWritable wordCount = new IntWritable(totalOccurrances);
		context.write(key, wordCount);
	}
}
