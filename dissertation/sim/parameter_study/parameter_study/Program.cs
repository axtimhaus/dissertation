using System.Text;

foreach (var arg in args)
    Console.Error.WriteLine(arg);

var input_file = args[0];
var output_file = args[1];

var input_text = File.ReadAllText(input_file, encoding: Encoding.UTF8);

File.WriteAllText(output_file, input_text, encoding: Encoding.UTF8);
