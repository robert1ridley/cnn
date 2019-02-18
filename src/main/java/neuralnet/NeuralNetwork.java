package neuralnet;

import java.lang.Math;
import java.util.HashMap;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;
import java.util.Random;
import java.util.Arrays;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.IOException;
public class NeuralNetwork{
//Map<String, String> dictionary = new HashMap<String, String>();
	public static double[][] w1;
	public static double[] b1;
	public static double[][] w2;
	public static double[] b2;
	public static double[] a1;
	public static double[] z1;
	public static double[] z2;
	public static double[] a2;
	public static double[] dz1;
	public static double[] dz2;
	public static double[][] dw1;
	public static double[][] dw2;
	public static double[] db1;
	public static double[] db2;

public static double[] tanh(double[] value) {
	double[] result=new double[value.length];
	for(int i=0;i<value.length;i++){
		double ex = Math.pow(Math.E, value[i]);// e^x
		double ey = Math.pow(Math.E, -value[i]);//e^(-x)
		double sinhx = ex-ey;
		double coshx = ex+ey;
	    result[i] = sinhx/coshx;
	}
	return result;
}
public static double[] power(double[] a){
	for(int i=0;i<a.length;i++)
		a[i]=a[i]*a[i];
	return a;
}
/* public static double[] tanhDerivative(double[] value){
	double[] result;
	for(int i=0;i<value.length;i++)
		result[i] = 1-tanh(value[i])*tanh(value[i]);
	return result;
} */

public static double[] log(double[] value,double t){
	double[] result=new double[value.length];
	for(int i=0;i<value.length;i++){
		result[i]=Math.log(value[i]+t);
		}
	return result;
}
public static double[] multiply(double[] value,double[] y) {
	double[] result=new double[value.length];
	for(int i=0;i<value.length;i++){
		result[i]=value[i]*y[i];
		}
	return result;
}
public static double[] multiply_plus(double t,double[] y) {
	double[] result=new double[y.length];
	for(int i=0;i<y.length;i++){
        result[i]=y[i]+t;
		}
	return result;
}
public static double[] multiply_plus(double[] t,double[] y) {
	double result[]=new double[y.length];
	for(int i=0;i<y.length;i++){
		result[i]=y[i]+t[i];
	}
	return result;
}
public static double[][] multiply_plus22(double[][] t,double[][] y) {
		double result[][]=new double[t.length][t[0].length];
	for(int i=0;i<y.length;i++)
		for(int j=0;j<y[i].length;j++)
		result[i][j]=y[i][j]+t[i][j];
	return result;
}
public static double[] matrix_opposite(double[] a){
	double result[]=new double[a.length];
	for(int i=0;i<a.length;i++){
		result[i]=-a[i];
	}
	return result;
}
public static double[][] matrix_opposite2(double[][] a){
	double result[][]=new double[a.length][a[0].length];
	for(int i=0;i<a.length;i++)
		for(int j=0;j<a[i].length;j++)
		result[i][j]=-a[i][j];
	return result;
}
public static double[][] matrix_dot(double a[][], double b[][]) {
        if (a[0].length != b.length)
            return null;
        int y = a.length;
        int x = b[0].length;
        double c[][] = new double[y][x];
        for (int i = 0; i < y; i++)
            for (int j = 0; j < x; j++)
                for (int k = 0; k < b.length; k++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }
public static double[] matrix_dot21(double a[][], double b[]) {
	 int y = a.length;
        double c[] = new double[y];
        for (int i = 0; i < y; i++){
            for (int j = 0; j < b.length; j++)
                    c[i] += a[i][j] * b[j];
		}
        return c;
}
public static double[][] matrix_dot11(double[] a, double[] b){
	double[][] result=new double[a.length][b.length];
	for(int i=0;i<a.length;i++)
		for(int j=0; j<b.length;j++)
			 result[i][j]=a[i]*b[j];
	return result;
}
public static double[][] matrix_trasport(double[][] a) {
	double C[][]=new double[a[0].length][a.length];
    for (int i = 0; i < a.length; i++) {
         for (int j = 0; j < a[i].length; j++) {
               C[j][i] = a[i][j];
           }
    }
        return C;
}
public static double[] matrix_power(double[] a) {
	double C[]=new double[a.length];
    for (int i = 0; i < a.length; i++) {
               C[i] = a[i]*a[i];
    }
        return C;
}
public static double[] sigmoid(double[] x){
	double[] s=new double[x.length];
     for(int i=0;i<x.length;i++){
        s[i]=1/(1+Math.pow(Math.E,-x[i]));
	 }
    return s;
}
public static double softmax(double[] x){
     double max=0;
	 double lable=0;
	 for(int i=0;i<x.length;i++){
		 if(max<x[i]){
		 max=x[i];
		 lable=i;
		 }
	 }
	 return lable;
}
public static void forward_propagation(double[] X){
	z1=matrix_dot21(w1,X);
	z1=multiply_plus(z1,b1);
	a1=tanh(z1);
	z2=matrix_dot21(w2,a1);
	z2=multiply_plus(z2,b2);
	a2=sigmoid(z2);
}
public static void back_propagation(double[] X,double[] Y){
	//double[] dz2;
	dz2=new double[10];
	dz1=new double[32];
	db1=new double[32];
	db2=new double[10];
    for(int i=0;i<a2.length;i++){
		dz2[i]=a2[i]-Y[i];
	}
    dw2=matrix_dot11(dz2,a1); 
     
    //db2=dz2;
	for(int i=0;i<dz2.length;i++)
		db2[i]=dz2[i];
    dz1=multiply(matrix_dot21(matrix_trasport(w2),dz2),(multiply_plus(1,matrix_opposite(matrix_power(a1))))); 
  
    dw1=matrix_dot11(dz1,X);
	for(int j=0;j<dz1.length;j++)
		db1[j]=dz1[j];
    //db1=dz1;
	
}
public static double[] multiply_shuc(double learning_rate,double[] a){
	for(int i=0;i<a.length;i++){
		a[i]=a[i]*learning_rate;
	}
	return a;
}
public static double[][] multiply_shuc2(double learning_rate,double[][] a){
	for(int i=0;i<a.length;i++)
		for(int j=0;j<a[i].length;j++)
			a[i][j]=a[i][j]*learning_rate;
	return a;
}
public static void update_para(double learning_rate ){
    w1=multiply_plus22(w1,matrix_opposite2(multiply_shuc2(learning_rate,dw1)));  
    b1=multiply_plus(b1,matrix_opposite(multiply_shuc(learning_rate,db1)));
	//System.out.println(Arrays.toString(w1[0]));
    w2=multiply_plus22(w2,matrix_opposite2(multiply_shuc2(learning_rate,dw2)));
    b2=multiply_plus(b2,matrix_opposite(multiply_shuc(learning_rate,db2)));
	for(int i=0;i<dw1.length;i++)
		for(int j=0;j<dw1[i].length;j++){
			dw1[i][j]=0;
		}
	for(int i=0;i<dw2.length;i++)
		for(int j=0;j<dw2[i].length;j++){
			dw2[i][j]=0;
		}
	for(int i=0;i<db1.length;i++){
			db1[i]=0;
			z1[i]=0;
			dz1[i]=0;
			a1[i]=0;
	}
	for(int i=0;i<db2.length;i++){
			db2[i]=0;
			z2[i]=0;
			dz2[i]=0;
			a2[i]=0;
			}
}
public double costloss(double[] Y){
	double t=0.00000000001;
    double sum=0;
	double[] logprobs=multiply_plus(multiply(log(a2,t),Y),multiply(log(matrix_opposite(a2),(1+t)),multiply_plus(1,matrix_opposite(Y))));
	for(int i=0;i<logprobs.length;i++){
		sum=sum+logprobs[i];
	}
	double cost=sum/a2.length;
    return cost;
}
public void initialize_with_zeros(int n_x,int n_h,int n_y){
	//double d =2*0.08575*Math.random()-0.08575;
	b1=new double[32];
	b2=new double[10];
	w1=new double[32][784];
	w2=new double[10][32];
	z1=new double[32];
	a1=new double[32];
	z2=new double[10];
	a2=new double[10];
	Random rand=new Random(2);
/* 	File file1 = new File("w1.txt");
	File file2 = new File("w2.txt"); */
/* 	if(!file1.exists()) {
			System.exit(0);
		}
	if(!file2.exists()) {
			System.exit(0);
		} */
	try{
	//Scanner s1 =new Scanner(file1);
	//Scanner s2 =new Scanner(file2);
		for(int i=0;i<n_y;i++)
		for(int j=0;j<n_h;j++){
			w2[i][j]=2*0.08575*rand.nextDouble()-0.08575;
		}//w2
			for(int i=0;i<n_h;i++)
		for(int j=0;j<n_x;j++){
			w1[i][j]=2*0.08575*rand.nextDouble()-0.08575;
		}
		}catch(Exception e){
            System.out.println("Wrong!");
        }
	for(int i=0;i<32;i++){
		b1[i]=0;	
	}//b1
	for(int i=0;i<10;i++){
		b2[i]=0;	
	}//b2

	//w1  Math.random  0.08575
}
public double[] image2vector(double[][] a){
	double[] b=new double[a.length*a[0].length];
	for(int i=0;i<a.length;i++)
		for(int j=0;j<a[0].length;j++)
			b[i*a.length+j]=a[i][j];
	return b;
}//
public double[] narmalize_data(double[] a)
{
	double max=0;
	double min=255;
	for(int i=0;i<a.length;i++){
		if(a[i]>max)
			max=a[i];
		if(a[i]<min)
			min=a[i];
	}
	for(int i=0;i<a.length;i++)
		a[i]=(a[i]-min)/(max-min);
	return a;
}//
public String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }

public double[][] getImages(String fileName) {
    double[][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);       
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);          
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);          
                x = new double[number][xPixel * yPixel];
                for (int i = 0; i < number; i++) {
                    double[] element = new double[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {
                        element[j] = bin.read();                            
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }
public double[] getLabels(String fileName) {
        double[] y = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                y = new double[number];
                for (int i = 0; i < number; i++) {
                    y[i] = bin.read();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }

public static void main(String[] args) throws IOException {
    String TRAIN_IMAGES_FILE = "train-images.idx3-ubyte";
    String TRAIN_LABELS_FILE = "train-labels.idx1-ubyte";
    String TEST_IMAGES_FILE = "t10k-images.idx3-ubyte";
    String TEST_LABELS_FILE = "t10k-labels.idx1-ubyte";
	NeuralNetwork dp=new NeuralNetwork();
	double[][] img_train = dp.getImages(TRAIN_IMAGES_FILE);  
    double[] train_labels = dp.getLabels(TRAIN_LABELS_FILE);   

    double[][] img_test = dp.getImages(TEST_IMAGES_FILE);
    double[] labels_test = dp.getLabels(TEST_LABELS_FILE);
	int ii=0;
    int n_x=784;
    int n_h=32;
    int n_y=10;
	dp.initialize_with_zeros(n_x,n_h,n_y);
	for(int i =0;i<60000;i++){
        double label_train[]=new double[10];
		for(int i1=0;i1<10;i1++){
			label_train[i1]=0;
		}
		double ttt=0.001;
        if(i>10){  
		    ttt=ttt*0.999;
			}
        label_train[(int)train_labels[i]]=1.0;
        double[] imgvector=dp.narmalize_data(img_train[i]);  
        dp.forward_propagation(imgvector);
        double costl=dp.costloss(label_train);  
        dp.back_propagation(imgvector, label_train); 
        dp.update_para(ttt);
    }
	
//	System.out.println(Arrays.toString(a1));
//	System.out.println(Arrays.toString(z1));
//	System.out.println(Arrays.toString(z2));
//	System.out.println(Arrays.toString(a2));
//	System.out.println(Arrays.toString(dz1));
//	System.out.println(Arrays.toString(dz2));
	
	
	for(int t_i=0;t_i<10000;t_i++){
		double[] img_train1=img_test[t_i];
		double[] vector_image=dp.narmalize_data(img_train1);
		double label_trainx=labels_test[t_i];
		dp.forward_propagation(vector_image);
		//System.out.println("nihao");
		double predict_value=softmax(a2);
		if((int)predict_value==(int)label_trainx){
			ii++;
		}
		//System.out.println(w1[0][0]);
	}
	System.out.println(ii);

  }
}
