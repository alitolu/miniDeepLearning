using System;
using System.Collections.Generic;
using System.Linq;

namespace MiniDeepLearning
{

    public class Tensor
    {
        public double[,] Data;
        public double[,] Grad;

        public int Rows => Data.GetLength(0);
        public int Cols => Data.GetLength(1);

        private static readonly Random Rand = new Random();

        public Tensor(int r, int c, bool random = false)
        {
            Data = new double[r, c];
            Grad = new double[r, c];

            if (random)
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < c; j++)
                        Data[i, j] = Rand.NextDouble() * 2 - 1;
        }

        public Tensor(double[,] data)
        {
            Data = data;
            Grad = new double[data.GetLength(0), data.GetLength(1)];
        }

        public void ZeroGrad()
        {
            Array.Clear(Grad, 0, Grad.Length);
        }

     
        public static Tensor MatMul(Tensor a, Tensor b)
        {
            var result = new Tensor(a.Rows, b.Cols);

            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < b.Cols; j++)
                    for (int k = 0; k < a.Cols; k++)
                        result.Data[i, j] += a.Data[i, k] * b.Data[k, j];

            return result;
        }

        public static Tensor Add(Tensor a, Tensor b)
        {
            var r = new Tensor(a.Rows, a.Cols);

            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    r.Data[i, j] = a.Data[i, j] + b.Data[i, j];

            return r;
        }

        public Tensor Apply(Func<double, double> f)
        {
            var r = new Tensor(Rows, Cols);

            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    r.Data[i, j] = f(Data[i, j]);

            return r;
        }
    }


    public abstract class Layer
    {
        public abstract Tensor Forward(Tensor x);
        public abstract Tensor Backward(Tensor gradOutput);
        public abstract IEnumerable<Tensor> Parameters();
    }


    public class Linear : Layer
    {
        private Tensor W;
        private Tensor B;

        private Tensor lastInput;

        public Linear(int input, int output)
        {
            W = new Tensor(input, output, random: true);
            B = new Tensor(1, output, random: true);
        }

        public override Tensor Forward(Tensor x)
        {
            lastInput = x;

            var y = Tensor.MatMul(x, W);

            for (int i = 0; i < y.Rows; i++)
                for (int j = 0; j < y.Cols; j++)
                    y.Data[i, j] += B.Data[0, j];

            return y;
        }

        public override Tensor Backward(Tensor gradOutput)
        {
            // dW = X^T * grad
            for (int i = 0; i < W.Rows; i++)
                for (int j = 0; j < W.Cols; j++)
                    for (int k = 0; k < lastInput.Rows; k++)
                        W.Grad[i, j] += lastInput.Data[k, i] * gradOutput.Data[k, j];

            // dB
            for (int j = 0; j < B.Cols; j++)
                for (int i = 0; i < gradOutput.Rows; i++)
                    B.Grad[0, j] += gradOutput.Data[i, j];

            // dInput = grad * W^T
            var gradInput = new Tensor(lastInput.Rows, W.Rows);

            for (int i = 0; i < gradInput.Rows; i++)
                for (int j = 0; j < gradInput.Cols; j++)
                    for (int k = 0; k < W.Cols; k++)
                        gradInput.Data[i, j] += gradOutput.Data[i, k] * W.Data[j, k];

            return gradInput;
        }

        public override IEnumerable<Tensor> Parameters()
        {
            yield return W;
            yield return B;
        }
    }


    public class Tanh : Layer
    {
        private Tensor lastOutput;

        public override Tensor Forward(Tensor x)
        {
            lastOutput = x.Apply(Math.Tanh);
            return lastOutput;
        }

        public override Tensor Backward(Tensor gradOutput)
        {
            var grad = new Tensor(lastOutput.Rows, lastOutput.Cols);

            for (int i = 0; i < grad.Rows; i++)
                for (int j = 0; j < grad.Cols; j++)
                {
                    double t = lastOutput.Data[i, j];
                    grad.Data[i, j] = (1 - t * t) * gradOutput.Data[i, j];
                }

            return grad;
        }

        public override IEnumerable<Tensor> Parameters()
            => Enumerable.Empty<Tensor>();
    }

    public class Sequential
    {
        private List<Layer> layers = new();

        public Sequential(params Layer[] l)
        {
            layers.AddRange(l);
        }

        public Tensor Forward(Tensor x)
        {
            foreach (var layer in layers)
                x = layer.Forward(x);

            return x;
        }

        public void Backward(Tensor grad)
        {
            for (int i = layers.Count - 1; i >= 0; i--)
                grad = layers[i].Backward(grad);
        }

        public IEnumerable<Tensor> Parameters()
            => layers.SelectMany(l => l.Parameters());
    }

    public class Adam
    {
        private List<Tensor> parameters;
        private List<double[,]> m = new();
        private List<double[,]> v = new();

        double lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        int t = 0;

        public Adam(IEnumerable<Tensor> parameters, double lr = 0.01)
        {
            this.parameters = parameters.ToList();
            this.lr = lr;

            foreach (var p in this.parameters)
            {
                m.Add(new double[p.Rows, p.Cols]);
                v.Add(new double[p.Rows, p.Cols]);
            }
        }

        public void Step()
        {
            t++;

            for (int idx = 0; idx < parameters.Count; idx++)
            {
                var p = parameters[idx];

                for (int i = 0; i < p.Rows; i++)
                    for (int j = 0; j < p.Cols; j++)
                    {
                        double g = p.Grad[i, j];

                        m[idx][i, j] = beta1 * m[idx][i, j] + (1 - beta1) * g;
                        v[idx][i, j] = beta2 * v[idx][i, j] + (1 - beta2) * g * g;

                        double mhat = m[idx][i, j] / (1 - Math.Pow(beta1, t));
                        double vhat = v[idx][i, j] / (1 - Math.Pow(beta2, t));

                        p.Data[i, j] -= lr * mhat / (Math.Sqrt(vhat) + eps);
                    }
            }
        }
    }


    public static class Loss
    {
        public static (double, Tensor) MSE(Tensor pred, Tensor target)
        {
            double loss = 0;
            var grad = new Tensor(pred.Rows, pred.Cols);

            for (int i = 0; i < pred.Rows; i++)
                for (int j = 0; j < pred.Cols; j++)
                {
                    double diff = pred.Data[i, j] - target.Data[i, j];
                    loss += diff * diff;
                    grad.Data[i, j] = 2 * diff / pred.Rows;
                }

            return (loss / pred.Rows, grad);
        }
    }


    class Program
    {
        static void Main()
        {
            var model = new Sequential(
                new Linear(2, 4),
                new Tanh(),
                new Linear(4, 1)
            );

            var optimizer = new Adam(model.Parameters(), 0.03);

            var X = new Tensor(new double[,]
            {
                {0,0},
                {0,1},
                {1,0},
                {1,1}
            });

            var Y = new Tensor(new double[,]
            {
                {0},
                {1},
                {1},
                {0}
            });

            for (int epoch = 0; epoch < 3000; epoch++)
            {
                foreach (var p in model.Parameters())
                    p.ZeroGrad();

                var pred = model.Forward(X);

                var (loss, grad) = Loss.MSE(pred, Y);

                model.Backward(grad);

                optimizer.Step();

                if (epoch % 300 == 0)
                    Console.WriteLine($"Epoch {epoch} Loss={loss:F6}");
            }

            Console.WriteLine("\nPredictions:");

            var outp = model.Forward(X);

            for (int i = 0; i < outp.Rows; i++)
                Console.WriteLine($"{X.Data[i,0]}, {X.Data[i,1]} -> {outp.Data[i,0]:F4}");
        }
    }
}
