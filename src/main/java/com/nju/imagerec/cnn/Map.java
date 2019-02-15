package com.nju.imagerec.cnn;

import java.io.IOException;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;

public class Map extends Mapper<LongWritable, Text, Text, DoubleWritable>{
	int count = 0;
	double[][] img_train = new double [60000][784];
	double[] train_labels = new double [60000];
	
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException, NoSuchElementException {
		// 一行一行读数据 （每一行数据最后值是标签）
		String sample = value.toString();
		StringTokenizer stringTokenizer = new StringTokenizer(sample);

		int i = 0;
		while (stringTokenizer.hasMoreTokens()) {
			if (i == 784) {
				train_labels[count] = Double.parseDouble(stringTokenizer.nextToken());
			}
			else {
				String term = stringTokenizer.nextToken();
				img_train[count][i] = Double.parseDouble(term);
			}
			i++;
		}
		count ++;
	}
	
	public void cleanup(Context context) throws IOException, InterruptedException {
		double[][] imageTrain = new double[count][784];
		double[] trainLabels = new double[count];
		for (int i = 0; i<count; i++) {
			imageTrain[i] = img_train[i];
			trainLabels[i] = train_labels[i];
		}
	    int n_x=784;
	    int n_h=32;
	    int n_y=10;
	    NeuralNetwork neuralNetwork = new NeuralNetwork();
	    neuralNetwork.initialize_with_zeros(n_x,n_h,n_y);
		for(int i = 0;i<imageTrain.length;i++){
	        double label_train[]=new double[10];
			for(int i1=0;i1<10;i1++){
				label_train[i1]=0;
			}
			double ttt=0.001;
	        if(i>10){  
			    ttt=ttt*0.999;
				}
	        label_train[(int)trainLabels[i]]=1.0;
	        double[] imgvector=neuralNetwork.narmalize_data(imageTrain[i]);  
	        neuralNetwork.forward_propagation(imgvector);
	        double costl=neuralNetwork.costloss(label_train);  
	        neuralNetwork.back_propagation(imgvector, label_train); 
	        neuralNetwork.update_para(ttt);
	    }
		
		// 把权重传给 Reducer
		double[][]w1 = neuralNetwork.w1;
		for (Integer i = 0; i < w1.length; i++) {
			for (Integer j = 0; j<w1[0].length; j++) {
				context.write(new Text("w1" + "-" + i.toString() + "-" + j.toString()) , new DoubleWritable(w1[i][j]));
			}
		}
		
		double[][]w2 = neuralNetwork.w2;
		for (Integer i = 0; i < w2.length; i++) {
			for (Integer j = 0; j<w2[0].length; j++) {
				context.write(new Text("w2" + "-" + i.toString() + "-" + j.toString()) , new DoubleWritable(w2[i][j]));
			}
		}
		
		double [] b1 = neuralNetwork.b1;
		for (Integer i = 0; i < b1.length; i++) {
			context.write(new Text("b1" + "-" + i.toString()), new DoubleWritable(b1[i]));
		}
		
		double [] b2 = neuralNetwork.b2;
		for (Integer i = 0; i < b2.length; i++) {
			context.write(new Text("b2" + "-" + i.toString()), new DoubleWritable(b2[i]));
		}
		
	}
	
}
