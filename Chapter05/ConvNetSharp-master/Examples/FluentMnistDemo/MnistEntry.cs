namespace FluentMnistDemo;

public class MnistEntry
{
    public byte[] Image;

    public int Label;

    public override string ToString()
    {
        return "Label: " + this.Label;
    }
}