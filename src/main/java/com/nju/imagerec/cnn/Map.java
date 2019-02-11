package com.nju.imagerec.cnn;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.commons.lang3.RandomStringUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class Map extends Mapper<LongWritable, Text, Text, IntWritable>{
	
//	public void setup() {
//		
//	}
	
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String text = value.toString();
		StringTokenizer stringTokenizer = new StringTokenizer(text);
		
		Text word = new Text();
		int count = 0;
		while (stringTokenizer.hasMoreTokens()) {
			word.set(stringTokenizer.nextToken());
			count += 1;
		}
		IntWritable newValue = new IntWritable(count);
		String newKey = RandomStringUtils.randomAlphanumeric(20).toUpperCase();
		Text newKeyText = new Text(newKey);
		context.write(newKeyText, newValue);
	}
	
//	public void cleanup() {
//		context.write(newKeyText, newValue);
//	}
}
