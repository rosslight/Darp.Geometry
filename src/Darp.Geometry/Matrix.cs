using System.Numerics.Tensors;

namespace Darp.Geometry;

/// <summary>Operations on read-only matrix access.</summary>
public static class Matrix
{
    // Copying / conversion

    public static void Clone<TM1, TMD>(in TM1 s1, in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c];
    }

    public static double[,] ToArray<TM1>(in TM1 s1)
        where TM1 : IReadOnlyMatrixD<TM1>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);

        double[,] result = new double[s1.Rows, s1.Columns];

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            result[r, c] = s1Span[r, c];

        return result;
    }

    // Scalar operations / reductions

    public static double Norm<TM1>(in TM1 s1)
        where TM1 : IReadOnlyMatrixD<TM1>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        return TensorKernels.Norm(s1Span);
    }

    public static double SquaredNorm<TM1>(in TM1 s1)
        where TM1 : IReadOnlyMatrixD<TM1>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        return TensorKernels.SquaredNorm(s1Span);
    }

    public static double InnerProduct<TM1, TM2>(in TM1 s1, in TM2 s2)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TM2 : IReadOnlyMatrixD<TM2>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease s2Lease = s2.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s2Span);
        return TensorKernels.InnerProduct(s1Span, s2Span);
    }

    public static double Dot<TM1, TM2>(in TM1 s1, in TM2 s2)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TM2 : IReadOnlyMatrixD<TM2>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease s2Lease = s2.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s2Span);
        return TensorKernels.Dot(s1Span, s2Span);
    }

    public static double Determinant3x3<TM1>(in TM1 s1)
        where TM1 : IReadOnlyMatrixD<TM1>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        return TensorKernels.Determinant3x3(s1Span);
    }

    // Unary matrix operations

    public static void Normalized<TM1, TMD>(in TM1 s1, in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);

        double norm = TensorKernels.Norm(s1Span);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c] / norm;
    }

    public static void Scale<TM1, TMD>(in TM1 s1, double scalar, in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c] * scalar;
    }

    public static void Divide<TM1, TMD>(in TM1 s1, double scalar, in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c] / scalar;
    }

    // Binary matrix operations

    public static void Add<TM1, TM2, TMD>(in TM1 s1, in TM2 s2, in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TM2 : IReadOnlyMatrixD<TM2>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease s2Lease = s2.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s2Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);
        MatrixShape.RequireSameShape(in s1Span, in s2Span, target);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c] + s2Span[r, c];
    }

    public static void Subtract<TM1, TM2, TMD>(in TM1 s1, in TM2 s2, in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TM2 : IReadOnlyMatrixD<TM2>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease s2Lease = s2.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s2Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);
        MatrixShape.RequireSameShape(in s1Span, in s2Span, target);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c] - s2Span[r, c];
    }

    public static void Multiply<TM1, TM2, TMD>(in TM1 s1, in TM2 s2, in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TM2 : IReadOnlyMatrixD<TM2>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease s2Lease = s2.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s2Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c] * s2Span[r, c];
    }

    public static void Lerp<TM1, TM2, TMD>(
        in TM1 s1,
        in TM2 s2,
        double amount,
        in TMD destination)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TM2 : IReadOnlyMatrixD<TM2>
        where TMD : IMatrixD<TMD>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease s2Lease = s2.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s2Span);
        using TensorSpanLease targetLease = destination.GetTensorSpan(out TensorSpan<double> target);

        for (int c = 0; c < s1.Columns; c++)
        for (int r = 0; r < s1.Rows; r++)
            target[r, c] = s1Span[r, c] + (s2Span[r, c] - s1Span[r, c]) * amount;
    }

    // Vector-specific operations

    public static Vector3D Cross<TM1, TM2>(in TM1 s1, in TM2 s2)
        where TM1 : IReadOnlyMatrixD<TM1>
        where TM2 : IReadOnlyMatrixD<TM2>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);
        using TensorSpanLease s2Lease = s2.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s2Span);
        return TensorKernels.Cross(s1Span, s2Span);
    }

    // Formatting

    internal static string Format<TM1>(in TM1 s1)
        where TM1 : IReadOnlyMatrixD<TM1>
    {
        using TensorSpanLease s1Lease = s1.GetReadOnlyTensorSpan(out ReadOnlyTensorSpan<double> s1Span);

        string[] rows = new string[s1.Rows];

        for (int r = 0; r < s1.Rows; r++)
        {
            string[] row = new string[s1.Columns];

            for (int c = 0; c < s1.Columns; c++)
                row[c] = s1Span[r, c].ToString();

            rows[r] = $"[{string.Join(", ", row)}]";
        }

        return string.Join(Environment.NewLine, rows);
    }
}
