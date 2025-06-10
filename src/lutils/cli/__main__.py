from pathlib import Path
from typing import Any
import click
import toml
import random
@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Lutils CLI - A command line interface for Lutils."""
    ctx.ensure_object(dict)
    DEFAULT: dict[str, list[dict[str, str]]] = {
        "students": [
            {
                "name": "default_name",
                "SID": "default_sid",
                "sex": "female"
            },
            {
                "name": "default_name2",
                "SID": "default_sid2",
                "sex": "male"
            },
            {
                "name": "default_name3",
                "SID": "default_sid3",
                "sex": "female"
            }
        ]
    }
    ctx.obj['DEFAULT'] = DEFAULT

@cli.command()
@click.option('--path',"-p" ,default= "./name_list.toml", help="名单文件")
@click.option("--num", "-n", default=1, type=int, help="点名数量，不会大于总数")
@click.option("--sex", "-s", default="both", type=click.Choice(["male", "female", "both"]), help="性别筛选")
@click.pass_context
def pick(ctx: click.Context, path: str, num: int, sex: str) -> None:
    """Pick a random item from a list."""   

    file_path = Path(path).expanduser().resolve()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            name_list = toml.load(f)
    except FileNotFoundError:
        with open(file_path, 'w', encoding='utf-8') as f:
            toml.dump(ctx.obj['DEFAULT'], f)
        name_list = ctx.obj['DEFAULT']
    
    students: list[dict[str, str]] = name_list.get('students', [])
    if not students:
        click.echo("名单文件中没有学生信息，请检查格式")
        return

    # 性别筛选
    if sex != "both":
        students = [student for student in students if student.get("sex", "unknown") == sex]

    if not students:
        click.echo(f"没有找到性别为 {sex} 的学生")
        return

    picked_students = random.sample(students, min(num, len(students)))
    for lucky in picked_students:
        click.echo(f"随机选中: \n姓名：{lucky['name']} (学号: {lucky['SID']}) 性别: {lucky['sex']}")

@cli.command()
@click.pass_context
@click.option('--force', is_flag=False, help="强制覆盖已有模板")
def init(ctx: click.Context, force: bool):
    """新建一个模板"""
    click.echo("正在创建模板...")
    if not force and Path("./name_list.toml").exists():
        click.echo("模板已存在，使用 --force 选项覆盖")
        if click.confirm("是否覆盖已有模板？", default=False):
            click.echo("正在覆盖模板...")
        else:
            click.echo("已取消模板创建")
            return
    with open("./name_list.toml", 'w') as f:
        toml.dump(ctx.obj['DEFAULT'], f)
    click.echo("模板创建完成！")


@cli.command("import")
@click.option('--path', "-p", required=True, help="导入文件路径")
def import_data(path: str) -> None:
    """导入数据"""
    file_path = Path(path).expanduser().resolve()
    output_path = Path("./name_list.toml").expanduser().resolve()
    
    if not file_path.exists():
        click.echo(f"文件 {file_path} 不存在")
        return
    
    if file_path.suffix == ".json":
        import json
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data: list[dict[str,str]] = json.load(f)
        except json.JSONDecodeError as e:
            click.echo(f"JSON 文件格式错误: {e}")
            return        
        except UnicodeDecodeError:
            click.echo("文件编码错误，请确保文件是UTF-8编码")
            return

        # 转换学生数据格式
        new_students: list[dict[str, str]] = []
        for student_data in data:  
            # 显式类型转换，确保类型检查器理解这是一个字符串键的字典
            raw_data: dict[str, Any] = student_data 
            # 映射字段：gender -> sex, code -> SID
            converted_student: dict[str, str] = {
                "name": str(raw_data.get("name", "")),
                "SID": str(raw_data.get("code", "")),
                "sex": "male" if raw_data.get("gender") == "男" else "female"
            }
            new_students.append(converted_student)
        
        # 读取现有的TOML文件（如果存在）
        existing_students: list[dict[str, str]] = []
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = toml.load(f)
                    existing_students = existing_data.get('students', [])
                    click.echo(f"找到现有名单，包含 {len(existing_students)} 名学生")
            except Exception as e:
                click.echo(f"读取现有文件时出错: {e}")
                existing_students = []
        
        # 合并学生数据，导入的数据覆盖原有的同名记录
        merged_students: list[dict[str, str]] = []
        new_student_sids = {student['SID'] for student in new_students}
          # 保留原有数据中不在新导入数据中的学生
        for existing_student in existing_students:
            if existing_student.get('SID') not in new_student_sids:
                merged_students.append(existing_student)
        
        # 添加新导入的学生数据
        merged_students.extend(new_students)
        
        # 构建TOML数据结构
        toml_data: dict[str, list[dict[str, str]]] = {"students": merged_students}
        # 保存到TOML文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                toml.dump(toml_data, f)
            click.echo(f"成功导入 {len(new_students)} 名学生，总计 {len(merged_students)} 名学生保存到 {output_path}")
        except Exception as e:
            click.echo(f"保存文件时出错: {e}")
            return
    else:
        click.echo("目前只支持导入 ZZU 学生名单 JSON【list[dict[str,str]]】 格式的文件")
        return
if __name__ == "__main__":
    cli()
